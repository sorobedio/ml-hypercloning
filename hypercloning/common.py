#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

import os

import torch


def scale_linear_layer(layer: torch.nn.Linear, scaler: float):
    """
    Scales the parameters of 'layer' so that its output is multiplied by 'scaler'.

    Arguments:
        layer:
            Linear layer to be scaled.
        scaler:
            Value to multiply the layer output.

    Returns:
        None.
    """
    layer.weight.data *= scaler
    if layer.bias is not None:
        layer.bias.data *= scaler


def get_noise_with_snr(weight: torch.tensor, snr_db: float):
    """
    Gaussian noise to be added to 'weight' so that the signal-to-noise
    ratio becomes 'snr_db'.

    Arguments:
        weight:
            Signal tensor.
        snr_db:
            Signal-to-noise ratio in decibels.

    Returns:
        Noise tensor.
    """
    signal_power = torch.mean(weight**2)
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = torch.randn_like(weight)
    current_noise_power = torch.mean(noise**2)
    noise = noise * torch.sqrt(noise_power / current_noise_power)
    return noise.to(weight.dtype)


def add_noise(weight, block_shape, snr_db):
    """
    Repeatedly adds and subtracts noise to 'block_shape' blocks within 'weight'.

    The noise is applied in alternating blocks of 'block_shape'.
    Below are several illustrations:

    Examples 1 & 2, even repetition of columns:
    +-------+-------+        +-------+-------+
    |   W   |   W   |        | W+N1  | W-N1  |
    +-------+-------+   -->  +-------+-------+
    |   W   |   W   |        | W+N2  | W-N2  |
    +-------+-------+        +-------+-------+

    +-------+-------+-------+-------+        +-------+-------+-------+-------+
    |   W   |   W   |   W   |   W   |        | W+N1  | W-N1  | W+N2  | W-N2  |
    +-------+-------+-------+-------+   -->  +-------+-------+-------+-------+
    |   W   |   W   |   W   |   W   |        | W+N3  | W-N3  | W+N4  | W-N4  |
    +-------+-------+-------+-------+        +-------+-------+-------+-------+

    Example 3, odd repetition of columns:
    +-------+-------+-------+        +-------+-------+-------+
    |   W   |   W   |   W   |        | W+N1  | W-N1  |   W   |
    +-------+-------+-------+   -->  +-------+-------+-------+
    |   W   |   W   |   W   |        | W+N2  | W-N2  |   W   |
    +-------+-------+-------+        +-------+-------+-------+

    Arguments:
        weight:
            Signal tensor.
        block_shape:
            Shape of the block to which noise is added or subtracted.
        snr_db:
            Signal-to-noise ratio in decibels.

    Returns:
        Noisy weight.
    """
    assert weight.shape[0] % block_shape[0] == 0
    assert weight.shape[1] % block_shape[1] == 0
    n_repeat_0 = weight.shape[0] // block_shape[0]
    n_repeat_1 = weight.shape[1] // block_shape[1]
    if weight.ndim == 2:
        for n0 in range(n_repeat_0):
            start0 = n0 * block_shape[0]
            end0 = start0 + block_shape[0]
            for n1 in range(n_repeat_1 // 2):
                start1 = 2 * n1 * block_shape[1]
                end1 = start1 + block_shape[1]
                start2 = (2 * n1 + 1) * block_shape[1]
                end2 = start2 + block_shape[1]
                noise = get_noise_with_snr(weight[start0:end0, start1:end1], snr_db)
                weight[start0:end0, start1:end1] += noise
                weight[start0:end0, start2:end2] -= noise
        return weight
    else:
        for n0 in range(weight.shape[0]):
            weight[n0] = add_noise(weight[n0], block_shape[1:], snr_db)
        return weight


def clone_matrix(dst_weight_shape, src_weight, snr_db=None, normalize=True):
    """
    Clones a matrix from 'src_weight' into 'dst_weight_shape'.

    Arguments:
        dst_weight_shape:
            Shape of the destination matrix. Must divide
            src_weight.shape.
        src_weight:
            Source weight to be cloned.
        snr_db:
            Signal-to-noise ratio in case noise is to be added.
            Defaults to None (no noise added).
        normalize:
            If True, normalize the weight by the number of repetitions
            in the second dimension.

    Returns:
        Cloned matrix with shape 'dst_weight_shape'.
    """
    out_features_old, in_features_old = src_weight.shape
    out_features_new, in_features_new = dst_weight_shape
    assert out_features_new >= out_features_old
    assert out_features_new % out_features_old == 0
    assert in_features_new >= in_features_old
    assert (
        in_features_new % in_features_old == 0
    ), f"{in_features_new} does not divide {in_features_old}"
    n_repeat_0 = out_features_new // out_features_old
    n_repeat_1 = in_features_new // in_features_old

    dst_weight = src_weight.data.repeat(n_repeat_0, n_repeat_1)
    if normalize:
        dst_weight = dst_weight / n_repeat_1
    if snr_db is not None:
        dst_weight = add_noise(dst_weight, src_weight.shape, snr_db)
    return dst_weight


def clone_vector(dst_vector_shape, src_vector):
    """
    Clones a vector from 'src_vector' into 'dst_vector_shape'.

    Arguments:
        dst_vector_shape:
            Shape of the destination vector. Must divide src_vector.shape.
        src_vector:
            Source vector to be cloned.

    Returns:
        Cloned vector with shape 'dst_vector_shape'.
    """
    assert src_vector.shape[0] <= dst_vector_shape[0]
    assert dst_vector_shape[0] % src_vector.shape[0] == 0
    n_repeat = dst_vector_shape[0] // src_vector.shape[0]
    dst_vector = src_vector.repeat(n_repeat)
    return dst_vector


def clone_linear_layer(dst_layer, src_layer, snr_db=None):
    """
    Clones linear layer parameters from 'src_layer' into 'dst_layer'.

    Arguments:
        dst_layer:
            Destination linear layer.
        src_layer:
            Source pretrained linear layer.
        snr_db:
            Optional signal-to-noise ratio in decibels to be added to the weight parameters of the destination layer.

    Returns:
        None.
    """
    dst_layer.weight.data = clone_matrix(
        dst_layer.weight.shape, src_layer.weight.data, snr_db=snr_db
    )
    if src_layer.bias is not None:
        assert (
            dst_layer.bias is not None
        ), "source model has bias in its linear layers but destination model doesn't"
        dst_layer.bias.data = clone_vector(dst_layer.bias.shape, src_layer.bias.data)


def clone_layer_norm(dst_layer, src_layer):
    """
    Clones normalization layer parameters from 'src_layer' into 'dst_layer'.

    Arguments:
        dst_layer:
            Destination normalization layer.
        src_layer:
            Source pretrained normalization layer.

    Returns:
        None.
    """
    if src_layer.weight is None and src_layer.bias is None:
        assert dst_layer.weight is None and dst_layer.bias is None
        return
    assert (
        dst_layer.eps == src_layer.eps
    ), f"eps should be the same for source and destination layer-norms, \
        got {src_layer.eps} and {dst_layer.eps}"
    assert (
        dst_layer.elementwise_affine == src_layer.elementwise_affine
    ), f"elementwise_affine should be the same for source and destination \
        layer-norms, got {src_layer.elementwise_affine} and {dst_layer.elementwise_affine}"
    dst_layer.weight.data = clone_vector(dst_layer.weight.shape, src_layer.weight)
    dst_layer.bias.data = clone_vector(dst_layer.bias.shape, src_layer.bias.data)


def clone_rms_norm(dst_layer, src_layer):
    """
    Clones rms-normalization layer parameters from 'src_layer' into 'dst_layer'.

    Arguments:
        dst_layer:
            Destination rms-normalization layer.
        src_layer:
            Source pretrained rms-normalization layer.

    Returns:
        None.
    """
    dst_layer.weight.data = clone_vector(dst_layer.weight.shape, src_layer.weight)


def rename_config(
    config, embedding_dim_multiplier: int = 1, up_project_multiplier: int = 1
):
    """
    adjusts the model name according to 'embedding_dim_multiplier' and 'up_project_multiplier'
    Arguments:
        config:
            config to be modified.
        embedding_dim_multiplier:
            expansion ratio of embedding dimension.
        up_project_multiplier:
            expansion ratio of ffn layer.
    Returns:
        updated config.

    """
    if embedding_dim_multiplier > 1:
        config._name_or_path += f"-{embedding_dim_multiplier}xembedding"
    if up_project_multiplier > 1:
        config._name_or_path += f"-{up_project_multiplier}xffn"
    return config


class scaledLinear(torch.nn.Module):
    """
    Wrapper layer that scales the weights of a linear layer before applying
    the linear transformation. This layer is useful in cases that embedding
    and unembedding layers are tied together, where the unembedding layer
    needs its weight to be scaled due to cloning but embedding layer should
    not scale the weights.
    Arguments:
        layer:
            original linear layer.
        scaler:
            scaler value.
    """

    def __init__(self, layer, scaler):
        super().__init__()
        self.layer = layer
        self.scaler = scaler
        self.weight = self.layer.weight
        self.bias = self.layer.bias

    def forward(self, x):
        weight = self.layer.weight * self.scaler
        if self.layer.bias is not None:
            bias = self.layer.bias * self.scaler
        else:
            bias = None
        return torch.nn.functional.linear(x, weight, bias)
