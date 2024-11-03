#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import copy

import numpy as np
import torch
from transformers import LlamaForCausalLM

from hypercloning.common import (
    clone_layer_norm,
    clone_linear_layer,
    clone_matrix,
    clone_rms_norm,
    rename_config,
    scale_linear_layer,
    scaledLinear,
)
from hypercloning.gemma_cloning import clone_gemma_attention


def clone_llama(
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
        pass 'num_heads_multiplier' to 'clone_llama'"

    # Set the destination network config according to user requested expansion factors:
    config = copy.deepcopy(src_network.config)
    config.hidden_size = embedding_dim_multiplier * config.hidden_size
    config.intermediate_size = up_project_multiplier * config.intermediate_size
    if config.num_key_value_heads != 1:
        config.num_key_value_heads = (
            embedding_dim_multiplier * config.num_key_value_heads
        )
    config.num_attention_heads = embedding_dim_multiplier * config.num_attention_heads
    config.tie_word_embeddings = False
    # rename the config according to expansion factors
    config = rename_config(config, embedding_dim_multiplier, up_project_multiplier)

    # Make an instance of the destination network:
    dst_network = LlamaForCausalLM._from_config(config)

    # Note: Gemma multiplies the embedding tokens by sqrt(emb_dim). We should normalize by
    # 1/sqrt(embedding_dim_multiplier) to avoid a mismatch:

    dst_network.model.embed_tokens.weight.data = clone_matrix(
        dst_network.model.embed_tokens.weight.data.shape,
        src_network.model.embed_tokens.weight.data,
        normalize=False,
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
    clone_linear_layer(dst_network.lm_head, src_network.lm_head)
    return dst_network