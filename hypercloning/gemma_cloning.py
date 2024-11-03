#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import copy

import numpy as np
import torch
from transformers import Gemma2ForCausalLM, GemmaForCausalLM

from hypercloning.common import (add_noise, clone_layer_norm,
                                 clone_linear_layer, clone_matrix,
                                 clone_rms_norm, rename_config,
                                 scale_linear_layer, scaledLinear)


def clone_gemma_attention(dst_layer, src_layer, snr_db=None):
    """
    Clones the attention layer from 'src_layer' into 'dst_layer'.

    Arguments:
        dst_layer: Destination attention layer.
        src_layer: Source (pretrained) attention layer.
        snr_db: signal to noise ratio. Defaults to None
    Returns:
        None.
    """

    src_config = copy.deepcopy(src_layer.config)
    dst_config = copy.deepcopy(dst_layer.config)
    clone_gemma_qkv_layer(
        dst_layer.q_proj,
        src_layer.q_proj,
        dst_layer.num_heads,
        src_layer.num_heads,
        snr_db=snr_db,
    )
    clone_gemma_qkv_layer(
        dst_layer.k_proj,
        src_layer.k_proj,
        dst_layer.num_key_value_heads,
        src_layer.num_key_value_heads,
        snr_db=snr_db,
    )
    clone_gemma_qkv_layer(
        dst_layer.v_proj,
        src_layer.v_proj,
        dst_layer.num_key_value_heads,
        src_layer.num_key_value_heads,
        snr_db=snr_db,
    )
    clone_linear_layer(dst_layer.o_proj, src_layer.o_proj, snr_db=snr_db)
    return dst_layer


def clone_gemma_qkv_layer(
    dst_layer, src_layer, num_heads_dst, num_heads_src, snr_db=None
):
    """
    Clones 'src_weight' into a weight tensor with 'dst_weight_shape' to be
    used in the attention layer.

    Arguments:
        dst_layer:
            Destination layer.
        src_layer:
            Source layer.
        num_heads_dst:
            Number of attention heads in the destination layer.
        num_heads_src:
            Number of attention heads in the source layer.
        snr_db:
            Signal-to-noise ratio. Defaults to None.

    Returns:
        None
    """
    dst_layer.weight.data = clone_gemma_qkv_weight(
        dst_layer.weight.shape,
        src_layer.weight.data,
        num_heads_dst,
        num_heads_src,
        snr_db=snr_db,
    )
    if src_layer.bias is not None:
        dst_layer.bias.data = clone_gemma_qkv_bias(
            dst_layer.bias.shape,
            src_layer.bias.data,
            num_heads_dst,
            num_heads_src,
        )


def clone_gemma_qkv_bias(dst_bias_shape, src_bias, num_heads_dst, num_heads_src):
    """
    Clones 'src_bias' into a bias vector with 'dst_bias_shape' to be
    used in the attention layer.

    Arguments:
        dst_bias_shape:
            Shape of the bias tensor in the destination layer.
        src_bias:
            bias vector in the source layer.
        num_heads_dst:
            Number of attention heads in the destination layer.
        num_heads_src:
            Number of attention heads in the source layer.

    Returns:
        Cloned QKV bias.
    """
    source_qkv_dim = src_bias.shape[0]
    destination_qkv_dim = dst_bias_shape[0]
    n_repeat = destination_qkv_dim // source_qkv_dim
    dst_bias = src_bias.reshape(num_heads_src, source_qkv_dim // num_heads_src)
    n_repeat_heads = num_heads_dst // num_heads_src
    n_repeat_head_dim = n_repeat // n_repeat_heads
    dst_bias = dst_bias.repeat(n_repeat_heads, n_repeat_head_dim)
    dst_bias = dst_bias.reshape(destination_qkv_dim)
    return dst_bias


def clone_gemma_qkv_weight(
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

    source_embedding_dimension = src_weight.shape[1]
    destination_embedding_dimension = dst_weight_shape[1]
    source_qkv_dim = src_weight.shape[0]
    destination_qkv_dim = dst_weight_shape[0]
    n_repeat_in = destination_embedding_dimension // source_embedding_dimension
    n_repeat = destination_qkv_dim // source_qkv_dim
    dst_weight = src_weight.reshape(
        num_heads_src, source_qkv_dim // num_heads_src, source_embedding_dimension
    )
    block_shape = dst_weight.shape
    n_repeat_heads = num_heads_dst // num_heads_src
    n_repeat_head_dim = n_repeat // n_repeat_heads
    dst_weight = (
        dst_weight.repeat(n_repeat_heads, n_repeat_head_dim, n_repeat_in) / n_repeat_in
    )
    if snr_db is not None:
        dst_weight = add_noise(dst_weight, block_shape, snr_db)
    dst_weight = dst_weight.reshape(
        destination_qkv_dim, destination_embedding_dimension
    )  # (d_head, n_heads, e) --> #(d_head*n_heads, e)

    return dst_weight


def clone_gemma(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """
    Cloning function for the Gemma family.

    For arguments description, refer to hypercloning.cloneModel.

    Returns:
        Cloned Gemma model instance.
    """
    snr_db = kwargs.get("snr_db", None)
    num_heads_multiplier = kwargs.get("num_heads_multiplier", embedding_dim_multiplier)
    assert (
        num_heads_multiplier == embedding_dim_multiplier
    ), "head_dim expansion is not supported for Gemma. The number of heads will \
        be automatically computed based on embedding dimension expansion. Do not \
        pass 'num_heads_multiplier' to 'clone_gemma'"

    # Set the destination network config according to user requested expansion factors:
    config = copy.deepcopy(src_network.config)
    config.hidden_size = embedding_dim_multiplier * config.hidden_size
    config.intermediate_size = up_project_multiplier * config.intermediate_size
    if config.num_key_value_heads != 1:
        config.num_key_value_heads = (
            embedding_dim_multiplier * config.num_key_value_heads
        )
    config.num_attention_heads = embedding_dim_multiplier * config.num_attention_heads

    # rename the config according to expansion factors
    config = rename_config(config, embedding_dim_multiplier, up_project_multiplier)

    # Make an instance of the destination network:
    dst_network = GemmaForCausalLM._from_config(config)

    # Note: Gemma multiplies the embedding tokens by sqrt(emb_dim). We should normalize by
    # 1/sqrt(embedding_dim_multiplier) to avoid a mismatch:

    dst_network.model.embed_tokens.weight.data = (
        clone_matrix(
            dst_network.model.embed_tokens.weight.data.shape,
            src_network.model.embed_tokens.weight.data,
            normalize=False,
        )
        * 1.0
        / np.sqrt(embedding_dim_multiplier)
    )

    for dst_layer, src_layer in zip(dst_network.model.layers, src_network.model.layers):
        clone_rms_norm(dst_layer.input_layernorm, src_layer.input_layernorm)
        clone_rms_norm(
            dst_layer.post_attention_layernorm, src_layer.post_attention_layernorm
        )
        dst_layer.self_attn = clone_gemma_attention(
            dst_layer.self_attn, src_layer.self_attn, snr_db=snr_db
        )
        clone_linear_layer(
            dst_layer.mlp.gate_proj, src_layer.mlp.gate_proj, snr_db=snr_db
        )
        clone_linear_layer(dst_layer.mlp.up_proj, src_layer.mlp.up_proj, snr_db=snr_db)
        clone_linear_layer(
            dst_layer.mlp.down_proj, src_layer.mlp.down_proj, snr_db=snr_db
        )
    clone_rms_norm(dst_network.model.norm, src_network.model.norm)

    # Note: the unembedding layer is tied with the embedding layer. We need to divide the
    # weights of the unembedding layer by 'embedding_dim_multiplier' but the weights in the
    # embedding layer should not be divided. but note that the embedding weights were already
    # divided by 'sqrt(embedding_dim_multiplier)' for Embedding initialization at the begining
    # of this function. So we use a wrapper class around lm_head that divides the weights in
    # the unembedding forward function by 'sqrt(embedding_dim_multiplier)' one more time:

    if embedding_dim_multiplier > 1:
        dst_network.lm_head = scaledLinear(
            dst_network.lm_head, 1.0 / np.sqrt(embedding_dim_multiplier)
        )
    return dst_network


def clone_gemma2(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """
    Cloning function for the Gemma2 family.

    For arguments description, refer to hypercloning.cloneModel.

    Returns:
        Cloned Gemma model instance.
    """
    snr_db = kwargs.get("snr_db", None)
    num_heads_multiplier = kwargs.get("num_heads_multiplier", embedding_dim_multiplier)
    assert (
        num_heads_multiplier == embedding_dim_multiplier
    ), "head_dim expansion is not supported for Gemma. The number of heads will \
        be automatically computed based on embedding dimension expansion. Do not \
        pass 'num_heads_multiplier' to 'clone_gemma2'"

    # Set the destination network config according to user requested expansion factors:
    config = copy.deepcopy(src_network.config)
    config.hidden_size = embedding_dim_multiplier * config.hidden_size
    config.intermediate_size = up_project_multiplier * config.intermediate_size
    if config.num_key_value_heads != 1:
        config.num_key_value_heads = (
            embedding_dim_multiplier * config.num_key_value_heads
        )
    config.num_attention_heads = embedding_dim_multiplier * config.num_attention_heads

    # rename the config according to expansion factors
    config = rename_config(config, embedding_dim_multiplier, up_project_multiplier)

    # Make an instance of the destination network:
    dst_network = Gemma2ForCausalLM._from_config(config)

    # Note: Gemma multiplies the embedding tokens by sqrt(emb_dim). We should normalize by
    # 1/sqrt(embedding_dim_multiplier) to avoid a mismatch:

    dst_network.model.embed_tokens.weight.data = (
        clone_matrix(
            dst_network.model.embed_tokens.weight.data.shape,
            src_network.model.embed_tokens.weight.data,
            normalize=False,
        )
        * 1.0
        / np.sqrt(embedding_dim_multiplier)
    )

    for dst_layer, src_layer in zip(dst_network.model.layers, src_network.model.layers):
        clone_rms_norm(dst_layer.input_layernorm, src_layer.input_layernorm)
        clone_rms_norm(
            dst_layer.post_attention_layernorm, src_layer.post_attention_layernorm
        )
        clone_rms_norm(
            dst_layer.pre_feedforward_layernorm, src_layer.pre_feedforward_layernorm
        )
        clone_rms_norm(
            dst_layer.post_feedforward_layernorm, src_layer.post_feedforward_layernorm
        )
        dst_layer.self_attn = clone_gemma_attention(
            dst_layer.self_attn, src_layer.self_attn, snr_db=snr_db
        )
        clone_linear_layer(
            dst_layer.mlp.gate_proj, src_layer.mlp.gate_proj, snr_db=snr_db
        )
        clone_linear_layer(dst_layer.mlp.up_proj, src_layer.mlp.up_proj, snr_db=snr_db)
        clone_linear_layer(
            dst_layer.mlp.down_proj, src_layer.mlp.down_proj, snr_db=snr_db
        )
    clone_rms_norm(dst_network.model.norm, src_network.model.norm)

    # Note: the unembedding layer is tied with the embedding layer. We need to divide the
    # weights of the unembedding layer by 'embedding_dim_multiplier' but the weights in the
    # embedding layer should not be divided. but note that the embedding weights were already
    # divided by 'sqrt(embedding_dim_multiplier)' for Embedding initialization at the begining
    # of this function. So we use a wrapper class around lm_head that divides the weights in
    # the unembedding forward function by 'sqrt(embedding_dim_multiplier)' one more time:

    if embedding_dim_multiplier > 1:
        dst_network.lm_head = scaledLinear(
            dst_network.lm_head, 1.0 / np.sqrt(embedding_dim_multiplier)
        )
    return dst_network
