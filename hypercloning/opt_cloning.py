#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import copy

import numpy as np
import torch
from transformers import OPTForCausalLM

from hypercloning.common import (clone_layer_norm, clone_linear_layer,
                                 clone_matrix, rename_config,
                                 scale_linear_layer, scaledLinear)


def clone_positional_embedding_layer(dst_layer, src_layer):
    """
    Clones the parameters of positional embedding from 'src_layer' to 'dst_layer'.

    Arguments:
        dst_layer: Destination layer.
        src_layer: Source (pretrained) layer.

    Returns:
        None.
    """

    src_weight = src_layer.weight.data
    dst_weight_shape = dst_layer.weight.shape
    assert src_weight.shape[1] <= dst_weight_shape[1]
    assert src_weight.shape[0] == dst_weight_shape[0]
    assert dst_weight_shape[1] % src_weight.shape[1] == 0
    n_repeat = dst_weight_shape[1] // src_weight.shape[1]
    dst_layer.weight.data = src_weight.repeat(1, n_repeat)


def clone_opt(
    src_network,
    embedding_dim_multiplier: int = 1,
    up_project_multiplier: int = 1,
    **kwargs,
):
    """
    Cloning function for the OPT family.

    For arguments description, refer to hypercloning.cloneModel.

    Returns:
        Cloned OPT model instance.
    """

    # Check if user has specified num_heads_multiplier manually. If not, set it
    # to 'embedding_dim_multiplier'.
    num_heads_multiplier = kwargs.get("num_heads_multiplier", embedding_dim_multiplier)
    snr_db = kwargs.get("snr_db", None)
    # Set the destination network config according to user requested expansion factors:
    config = copy.deepcopy(src_network.config)
    setattr(config, "hidden_size", embedding_dim_multiplier * config.hidden_size)
    setattr(config, "ffn_dim", up_project_multiplier * config.ffn_dim)
    setattr(
        config, "num_attention_heads", num_heads_multiplier * config.num_attention_heads
    )

    old_head_dim = (
        src_network.config.hidden_size // src_network.config.num_attention_heads
    )
    new_head_dim = config.hidden_size // config.num_attention_heads

    # rename the config according to expansion factors
    config = rename_config(config, embedding_dim_multiplier, up_project_multiplier)

    # If head_dim changes, attention query and key layers should be scaled:
    attention_scaler = np.sqrt(np.sqrt(old_head_dim * 1.0 / new_head_dim))

    assert (
        new_head_dim % old_head_dim == 0
    ), f"new head dimension ({new_head_dim}) should divide original head dimension \
        ({old_head_dim}). Consider changing num_heads_multiplier ({num_heads_multiplier})"

    # Make an instance of the destination network:
    dst_network = OPTForCausalLM._from_config(config)

    # Set the embedding layer parameters:
    dst_network.model.decoder.embed_tokens.weight.data = clone_matrix(
        dst_network.model.decoder.embed_tokens.weight.data.shape,
        src_network.model.decoder.embed_tokens.weight.data,
        normalize=False,
    )

    # If the network uses 'project_in' and 'project_out', clone these parameters:
    if dst_network.model.decoder.project_in is not None:
        clone_linear_layer(
            dst_network.model.decoder.project_in,
            src_network.model.decoder.project_in,
            snr_db=snr_db,
        )
        clone_linear_layer(
            dst_network.model.decoder.project_out,
            src_network.model.decoder.project_out,
            snr_db=snr_db,
        )

    # Clone the positional embedding layer parameters:
    clone_positional_embedding_layer(
        dst_network.model.decoder.embed_positions,
        src_network.model.decoder.embed_positions,
    )

    # Clone the final layer norm if required:
    if src_network.model.decoder.final_layer_norm is not None:
        clone_layer_norm(
            dst_network.model.decoder.final_layer_norm,
            src_network.model.decoder.final_layer_norm,
        )

    # Iterate through the decoder layers and clone the components:
    for dst_layer, src_layer in zip(
        dst_network.model.decoder.layers, src_network.model.decoder.layers
    ):
        clone_linear_layer(
            dst_layer.self_attn.k_proj, src_layer.self_attn.k_proj, snr_db=snr_db
        )
        scale_linear_layer(dst_layer.self_attn.k_proj, attention_scaler)
        clone_linear_layer(
            dst_layer.self_attn.q_proj, src_layer.self_attn.q_proj, snr_db=snr_db
        )
        scale_linear_layer(dst_layer.self_attn.q_proj, attention_scaler)
        clone_linear_layer(
            dst_layer.self_attn.v_proj, src_layer.self_attn.v_proj, snr_db=snr_db
        )
        clone_linear_layer(
            dst_layer.self_attn.out_proj, src_layer.self_attn.out_proj, snr_db=snr_db
        )
        clone_layer_norm(dst_layer.self_attn_layer_norm, src_layer.self_attn_layer_norm)
        clone_linear_layer(dst_layer.fc1, src_layer.fc1, snr_db=snr_db)
        clone_linear_layer(dst_layer.fc2, src_layer.fc2, snr_db=snr_db)
        clone_layer_norm(dst_layer.final_layer_norm, src_layer.final_layer_norm)

    # Note: the unembedding layer is tied with the embedding layer. We need to divide the
    # weights of the unembedding layer by 'embedding_dim_multiplier' but the weights in the
    # embedding layer should not be divided. Therefore, we use a wrapper class around lm_head:
    if embedding_dim_multiplier > 1:
        dst_network.lm_head = scaledLinear(
            dst_network.lm_head, 1.0 / embedding_dim_multiplier
        )

    return dst_network
