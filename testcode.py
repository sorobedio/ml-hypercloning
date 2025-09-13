

from transformers import AutoModelForCausalLM
from ml-hypercloning.hypercloning import cloneModel



if __name__=='__main__':
    # instantiate the source model (pretrained):
    source_model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")

    # Clone a model with 2x embedding size and 2x FFN dimension:
    destination_model = cloneModel(source_model, embedding_dim_multiplier=4)

    # from transformers import OPTConfig, OPTModel
    #
    # # Initializing a OPT facebook/opt-large style configuration
    # configuration = OPTConfig()
    #
    # # Initializing a model (with random weights) from the facebook/opt-large style configuration
    # model = OPTModel(configuration)
    #
    # # Accessing the model configuration
    # configuration = model.config