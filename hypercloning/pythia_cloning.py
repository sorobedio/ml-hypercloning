#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import copy

import numpy as np
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTNeoXConfig,
    GPTNeoXForCausalLM,
)

from hypercloning.common import (
    add_noise,
    clone_layer_norm,
    clone_linear_layer,
    clone_matrix,
    rename_config,
)


def clone_pythia_qkv_layer(
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

    dst_layer.weight.data = clone_pythia_qkv_weight(
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
        dst_layer.bias.data = clone_pythia_qkv_bias(
            dst_layer.bias.shape, src_layer.bias.data, num_heads_dst, num_heads_src
        )


def clone_pythia_qkv_weight(
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
    if num_heads_dst // num_heads_src == n_repeat:
        assert (
            source_embedding_dim // num_heads_src
            == destination_embedding_dim // num_heads_dst
        ), "either number of heads or head-dims should be the same in source and \
            cloned network. Cannot change both!"
        dst_weight = src_weight.reshape(
            num_heads_src,
            3,
            source_embedding_dim // num_heads_src,
            source_embedding_dim,
        )
        block_shape = dst_weight.shape
        dst_weight = dst_weight.repeat(n_repeat, 1, 1, n_repeat) / n_repeat
    else:
        assert (
            num_heads_src == num_heads_dst
        ), "either number of heads or head-dims should be the same in source and \
            cloned network. Cannot change both!"
        dst_weight = src_weight.reshape(
            3,
            num_heads_src,
            source_embedding_dim // num_heads_src,
            source_embedding_dim,
        )
        dst_weight = dst_weight.repeat(1, 1, n_repeat, n_repeat) / n_repeat
        block_shape = dst_weight.shape
    if snr_db is not None:
        dst_weight = add_noise(dst_weight, block_shape, snr_db)
    dst_weight = dst_weight.reshape(
        3 * destination_embedding_dim, destination_embedding_dim
    )  # (3, n_heads, d_head, e) --> #(3*n_heads*d_head, e)

    return dst_weight


def clone_pythia_qkv_bias(dst_bias_shape, src_bias, num_heads_dst, num_heads_src):
    """
    Clones 'src_bias' into a bias tensor with 'dst_bias_shape' to be used
    in the attention layer.

    Arguments:
        dst_bias_shape:
            Shape of the bias tensor in the destination layer.
        src_bias:
            Bias tensor in the source layer.
        num_heads_dst:
            Number of attention heads in the destination layer.
        num_heads_src:
            Number of attention heads in the source layer.

    Returns:
        Cloned QKV bias.
    """

    source_embedding_dim = src_bias.shape[0] // 3
    destination_embedding_dim = dst_bias_shape[0] // 3
    n_repeat = destination_embedding_dim // source_embedding_dim
    if num_heads_dst // num_heads_src == n_repeat:
        dst_bias = src_bias.reshape(
            num_heads_src, 3, source_embedding_dim // num_heads_src
        )
        dst_bias = dst_bias.repeat(n_repeat, 1, 1)
    else:
        dst_bias = src_bias.reshape(
            3, num_heads_src, source_embedding_dim // num_heads_src
        )
        dst_bias = dst_bias.repeat(1, 1, n_repeat)
    return dst_bias.reshape(-1)


def clone_GPTNeoXAttention(dst_layer, src_layer, snr_db=None):
    """
    Clones the attention layer from 'src_layer' into 'dst_layer'.

    Arguments:
        dst_layer: Destination attention layer.
        src_layer: Source (pretrained) attention layer.

    Returns:
        None.
    """

    clone_pythia_qkv_layer(
        dst_layer.query_key_value,
        src_layer.query_key_value,
        dst_layer.num_attention_heads,
        src_layer.num_attention_heads,
        snr_db=snr_db,
    )
    clone_linear_layer(dst_layer.dense, src_layer.dense, snr_db=snr_db)


def clone_pythia(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """
    Clones Pythia models.

    For arguments description, refer to hypercloning.cloneModel.

    Returns:
        Cloned Pythia model instance.
    """
    num_heads_multiplier = kwargs.get("num_heads_multiplier", embedding_dim_multiplier)
    snr_db = kwargs.get("snr_db", None)
    assert (
        num_heads_multiplier == embedding_dim_multiplier
    ), "head_dim expansion is not supported for Pythia. The number of heads will \
        be automatically computed based on ebedding dimension expansion. Do not \
        pass 'num_heads_multiplier' to 'clone_pythia'"
    config = GPTNeoXConfig(**src_network.config.to_dict())

    # Set the new config parameters for the destination (expanded) network:
    config.hidden_size = embedding_dim_multiplier * config.hidden_size
    config.num_attention_heads = num_heads_multiplier * config.num_attention_heads
    old_head_dim = (
        src_network.config.hidden_size // src_network.config.num_attention_heads
    )
    new_head_dim = config.hidden_size // config.num_attention_heads
    config.intermediate_size = up_project_multiplier * config.intermediate_size

    # rename the config according to expansion factors
    config = rename_config(config, embedding_dim_multiplier, up_project_multiplier)

    # Create the destination network:
    dst_network = GPTNeoXForCausalLM._from_config(
        config,
        torch_dtype=torch.bfloat16,
        attn_implementation=src_network.config._attn_implementation,
    )

    # Clone the embedding layer parameters
    dst_network.gpt_neox.embed_in.weight.data = clone_matrix(
        dst_network.gpt_neox.embed_in.weight.data.shape,
        src_network.gpt_neox.embed_in.weight.data,
        normalize=False,
    )

    # Iterate through pairs of layers in source and destination layers.
    # Clone source layer components into destination layer components:
    for dst_layer, src_layer in zip(
        dst_network.gpt_neox.layers, src_network.gpt_neox.layers
    ):
        clone_layer_norm(dst_layer.input_layernorm, src_layer.input_layernorm)
        clone_layer_norm(
            dst_layer.post_attention_layernorm, src_layer.post_attention_layernorm
        )
        clone_GPTNeoXAttention(dst_layer.attention, src_layer.attention, snr_db=snr_db)
        clone_linear_layer(
            dst_layer.mlp.dense_h_to_4h, src_layer.mlp.dense_h_to_4h, snr_db=snr_db
        )
        clone_linear_layer(
            dst_layer.mlp.dense_4h_to_h, src_layer.mlp.dense_4h_to_h, snr_db=snr_db
        )

    # Clone the final layer norm:
    clone_layer_norm(
        dst_network.gpt_neox.final_layer_norm, src_network.gpt_neox.final_layer_norm
    )

    # Clone the unembedding layer:
    clone_linear_layer(dst_network.embed_out, src_network.embed_out)
    return dst_network