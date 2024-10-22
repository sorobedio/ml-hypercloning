#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import copy

import numpy as np
import torch
from transformers import AutoModelForCausalLM

from hypercloning.common import (add_noise, clone_layer_norm,
                                 clone_linear_layer, clone_matrix,
                                 rename_config, scale_linear_layer,
                                 scaledLinear)


class clonedPositionalEmbedding(torch.nn.Module):
    """
    Clones a source positional embedding layer called 'emb_module'.

    This function assumes 'emb_module' receives an input 'x' and computes P(x).
    The cloned module takes repeated inputs [x, ..., x]^T and computes the
    output as [P(x), ..., P(x)]^T.

    Arguments:
        emb_module:
            Original positional embedding module from the source network.
        repeat:
            Expansion factor indicating how many times the module should
            be called repeatedly.

    """

    def __init__(self, emb_module, repeat):
        super().__init__()
        self.repeat = repeat
        self.module = emb_module

    def forward(self, q, k):
        """
        shape of q: (...D) where D should be split.
        shape of k: (...D) where D should be split.
        """
        assert (
            q.shape[-1] % self.repeat
        ) == 0, (
            f"q tensor dimension ({q.shape}) does not divide repeats ({self.repeat})"
        )
        assert (
            k.shape[-1] % self.repeat
        ) == 0, (
            f"k tensor dimension ({k.shape}) does not divide repeats ({self.repeat})"
        )
        q_split = torch.split(q, q.shape[-1] // self.repeat, dim=-1)
        k_split = torch.split(k, k.shape[-1] // self.repeat, dim=-1)
        outputs = [self.module(qq, kk) for qq, kk in zip(q_split, k_split)]
        qs = [o[0] for o in outputs]
        ks = [o[1] for o in outputs]
        return torch.cat(qs, dim=-1), torch.cat(ks, dim=-1)


def reorder_swiglu(tensor, n_repeat):
    """
    Reorders the rows in the first linear layer of the FFN to correct
    the gating that occurs in the subsequent SWIGLU activation function.

    The original linear layer produces the output [x_top, x_bottom]^T,
    and the following SWIGLU computes the activation as 'silu(x_top) * x_bottom'.

    In contrast, the cloned layer produces the output [x_up, x_down, x_up, x_down]^T.
    The SWIGLU incorrectly computes 'silu([x_top, x_bottom]^T) * [x_top, x_bottom]^T',
    which is not the desired behavior.

    To fix this, we need to reorder the weights of the cloned linear layer
    so that it produces [x_up, x_up, x_down, x_down]^T. This allows the SWIGLU
    to correctly compute 'silu([x_top, x_top]^T) * [x_bottom, x_bottom]^T'.

    Arguments:
        tensor:
            The weight or bias tensor in the destination FFN block.
        n_repeat:
            The expansion factor from the source to the destination FFN block.

    Returns:
        Reordered tensor.
    """
    n = int(tensor.shape[0] // (n_repeat * 2))
    tensors_up = [tensor[2 * i * n : (2 * i + 1) * n] for i in range(0, n_repeat)]
    tensors_down = [
        tensor[(2 * i + 1) * n : (2 * i + 2) * n] for i in range(0, n_repeat)
    ]
    all_tensors = tensors_up + tensors_down
    return torch.cat(all_tensors, dim=0)


def reorder_weights(w, n_heads_old, head_dim_old, n_repeat_dim, n_repeat_heads):
    """
    Reorders the columns of the out_project linear layer at the end of the
    attention layer.

    This function is meant to preserve the functionality of the attention
    block when the head_dim is changed.

    Arguments:
        w:
            The weight tensor to be reordered (from the o_proj linear layer).
        n_heads_old:
            Number of heads in the source attention layer.
        head_dim_old:
            Dimension of each head in the source attention layer.
        n_repeat_dim:
            Number of times the head-dim of the source attention layer is
            repeated in the destination attention layer.
        n_repeat_heads:
            Number of times the heads of the source attention layer are
            repeated in the destination attention layer.

    Returns:
        Reordered weights.
    """
    w = w.reshape(w.shape[0], n_repeat_heads, n_repeat_dim, n_heads_old, head_dim_old)
    sh_old = copy.copy(w.shape)
    w = w.permute(0, 1, 3, 2, 4)
    sh_new = copy.copy(w.shape)
    w = w.reshape(w.shape[0], -1)
    return w


def clone_olmo_qkv_layer(
    dst_layer, src_layer, num_heads_dst, num_heads_src, snr_db=None
):
    """
    Clones a source QKV linear layer into a destination QKV linear layer.

    Arguments:
        dst_layer: Destination layer.
        src_layer: Source layer.
        num_heads_dst: Number of attention heads in the destination layer.
        num_heads_src: Number of attention heads in the source layer.
        snr_db: Signal-to-noise ratio. Defaults to None.

    Returns:
        None.
    """

    dst_layer.weight.data = clone_olmo_qkv_weight(
        dst_layer.weight.shape,
        src_layer.weight.data,
        num_heads_dst,
        num_heads_src,
        snr_db=snr_db,
    )
    if src_layer.bias is not None:
        assert (
            dst_layer.bias is not None
        ), "source model has bias in it's linear layers but destination model doesn't"
        dst_layer.bias.data = clone_olmo_qkv_bias(
            dst_layer.bias.shape, src_layer.bias.data, num_heads_dst, num_heads_src
        )


def clone_olmo_qkv_weight(
    dst_weight_shape, src_weight, num_heads_dst, num_heads_src, snr_db=None
):
    """
    Clones 'src_weight' into a weight tensor with 'dst_weight_shape' to be
    used in the attention layer.

    Arguments:
        dst_weight_shape:
            Shape of the weight tensor in the destination layer.
        src_weight:
            Weight tensor in the source layer.
        num_heads_dst:
            Number of attention heads in the destination layer.
        num_heads_src:
            Number of attention heads in the source layer.
        snr_db:
            Signal-to-noise ratio. Defaults to None.

    Returns:
        Cloned QKV weights.
    """
    assert src_weight.shape[0] == (3 * src_weight.shape[1])
    assert dst_weight_shape[0] == (3 * dst_weight_shape[1])
    source_embedding_dim = src_weight.shape[1]
    destination_embedding_dim = dst_weight_shape[1]
    n_repeat = destination_embedding_dim // source_embedding_dim
    dst_weight = src_weight.reshape(
        3, num_heads_src, source_embedding_dim // num_heads_src, source_embedding_dim
    )  # (3, H, E/H, E)
    block_shape = dst_weight.shape
    head_repeat = num_heads_dst // num_heads_src
    dim_repeat = n_repeat // head_repeat
    dst_weight = (
        dst_weight.repeat(1, head_repeat, dim_repeat, n_repeat) / n_repeat
    )  # (3, nH, E/H, nE)
    dst_weight[:2] = dst_weight[:2] / np.sqrt(
        np.sqrt(dim_repeat)
    )  ##divide query and key weights to compensate for normalization
    if snr_db is not None:
        dst_weight = add_noise(dst_weight, block_shape, snr_db)
    dst_weight = dst_weight.reshape(
        3 * destination_embedding_dim, destination_embedding_dim
    )  # (3, n_heads, d_head, e) --> #(3*n_heads*d_head, e)

    return dst_weight


def clone_olmo_qkv_bias(dst_bias_shape, src_bias, num_heads_dst, num_heads_src):
    assert False, "not implemented"


def clone_olmo(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """
    Clones the OLMo network. See hypercloning.cloneModel for argument descriptions.

    Returns:
        Cloned OLMo network instance.
    """

    # Check if user has specified num_heads_multiplier manually. If not, set it
    # to 'embedding_dim_multiplier'.
    num_heads_multiplier = kwargs.get("num_heads_multiplier", embedding_dim_multiplier)
    snr_db = kwargs.get("snr_db", None)
    # Set the destination network config according to user requested expansion factors:
    config = copy.deepcopy(src_network.config)
    setattr(config, "d_model", embedding_dim_multiplier * config.d_model)
    if getattr(config, "mlp_hidden_size", None) is None:
        setattr(
            config,
            "mlp_hidden_size",
            src_network.config.d_model * src_network.config.mlp_ratio,
        )
    setattr(config, "mlp_hidden_size", up_project_multiplier * config.mlp_hidden_size)
    setattr(config, "n_heads", num_heads_multiplier * config.n_heads)

    # rename the config according to expansion factors
    config = rename_config(config, embedding_dim_multiplier, up_project_multiplier)

    # lazy approach: disable weight tying to avoid mismatch
    config.weight_tying = False

    old_head_dim = src_network.config.d_model // src_network.config.n_heads
    new_head_dim = config.d_model // config.n_heads

    assert (
        new_head_dim % old_head_dim == 0
    ), f"new head dimension ({new_head_dim}) should divide original head \
        dimension ({old_head_dim}). Consider changing num_heads_multiplier \
        ({num_heads_multiplier})"
    dst_network = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, torch_dtype=torch.bfloat16
    )

    # Clone the embedding layer:
    dst_network.model.transformer.wte.weight.data = clone_matrix(
        dst_network.model.transformer.wte.weight.data.shape,
        src_network.model.transformer.wte.weight.data,
        normalize=False,
    )

    # Clone layernorm:
    clone_layer_norm(
        dst_network.model.transformer.ln_f, src_network.model.transformer.ln_f
    )

    # Iterate through decoder layers and clone their components:
    for dst_layer, src_layer in zip(
        dst_network.model.transformer.blocks, src_network.model.transformer.blocks
    ):
        clone_linear_layer(dst_layer.attn_out, src_layer.attn_out, snr_db=snr_db)

        # If head_dim has changed, we need to re-order the weights of dst_layer.attn_out
        # to match the expansion in attention heads:
        if new_head_dim != old_head_dim:
            dst_layer.attn_out.weight.data = reorder_weights(
                w=dst_layer.attn_out.weight.data,
                n_heads_old=src_layer.config.n_heads,
                head_dim_old=old_head_dim,
                n_repeat_dim=new_head_dim // old_head_dim,
                n_repeat_heads=num_heads_multiplier,
            )

        clone_linear_layer(dst_layer.ff_out, src_layer.ff_out, snr_db=snr_db)
        clone_layer_norm(dst_layer.attn_norm, src_layer.attn_norm)
        clone_layer_norm(dst_layer.ff_norm, src_layer.ff_norm)
        clone_olmo_qkv_layer(
            dst_layer.att_proj,
            src_layer.att_proj,
            dst_layer.config.n_heads,
            src_layer.config.n_heads,
            snr_db=snr_db,
        )
        clone_linear_layer(dst_layer.ff_proj, src_layer.ff_proj, snr_db=snr_db)

        # If the input to SWIGLU activaion has changed dimension, we should reorder weights:
        swiglu_repeat = dst_layer.ff_proj.out_features // (
            src_layer.ff_proj.out_features
        )
        if swiglu_repeat > 1:
            dst_layer.ff_proj.weight.data = reorder_swiglu(
                dst_layer.ff_proj.weight.data, swiglu_repeat
            )
            if dst_layer.ff_proj.bias is not None:
                dst_layer.ff_proj.bias.data = reorder_swiglu(
                    dst_layer.ff_proj.bias.data, swiglu_repeat
                )

        # If the head dimension has changed, we should fix positional embedding:
        if new_head_dim > old_head_dim:
            dst_layer.rotary_emb = clonedPositionalEmbedding(
                src_layer.rotary_emb, new_head_dim // old_head_dim
            )

    if not src_network.config.weight_tying:
        # Clone the unembedding layer  from the source network unembedding:
        dst_network.model.transformer.ff_out.weight.data = clone_matrix(
            dst_network.model.transformer.ff_out.weight.data.shape,
            src_network.model.transformer.ff_out.weight.data,
            normalize=True,
        )
    else:
        # Clone the unembedding layer from the source network embedding layer:
        dst_network.model.transformer.ff_out.weight.data = clone_matrix(
            dst_network.model.transformer.ff_out.weight.data.shape,
            src_network.model.transformer.wte.weight.data,
            normalize=True,
        )
    return dst_network
