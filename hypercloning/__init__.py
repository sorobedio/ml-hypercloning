#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2020 Apple Inc. All Rights Reserved.
#

from hf_olmo.configuration_olmo import OLMoConfig
from transformers import GemmaConfig, Gemma2Config, GPTNeoXConfig, LlamaConfig, OPTConfig

from hypercloning.gemma_cloning import clone_gemma, clone_gemma2
from hypercloning.llama_cloning import clone_llama
from hypercloning.olmo_cloning import clone_olmo
from hypercloning.opt_cloning import clone_opt
from hypercloning.pythia_cloning import clone_pythia

REGISTERED_CLONING_FUNCTIONS = {
    "LlamaConfig": clone_llama,
    "GemmaConfig": clone_gemma,
    "Gemma2Config": clone_gemma2,
    "OPTConfig": clone_opt,
    "OLMoConfig": clone_olmo,
    "GPTNeoXConfig": clone_pythia,
}

def cloneModel(
    model, embedding_dim_multiplier: int, up_project_multiplier: int, **kwargs
):
    """
    Expand 'model' according to 'embedding_dim_multiplier' and
    'up_project_multiplier'.

    Arguments:
        embedding_dim_multiplier:
            Expansion factor for embedding size.
        up_project_multiplier:
            Expansion factor for the FFN layers.
        kwargs can include:
            snr_db:
                Signal to noise ratio in decibels if noise is desired to be
                added to the weight tensors. Defaults to None.
            up_project_multiplier:
                The ratio of the number of heads in the destination network
                divided by the number of heads in the source network.
                Defaults to 'embedding_dim_multiplier' (recommended).

    Returns:
        Cloned model with expanded parameters.
    """
    cloning_function_key = str(type(model.config)).split(".")[-1][:-2].strip()
    
    assert (
        cloning_function_key in REGISTERED_CLONING_FUNCTIONS
    ), f"cloning is not supported for model config of type {cloning_function_key}"
    cloning_function = REGISTERED_CLONING_FUNCTIONS[cloning_function_key]
    print(f"cloning the network using {cloning_function} ...")
    return cloning_function(
        model,
        embedding_dim_multiplier=embedding_dim_multiplier,
        up_project_multiplier=up_project_multiplier,
        **kwargs,
    )
    