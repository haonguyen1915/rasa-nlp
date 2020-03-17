from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.components import ComponentBuilder
from rasa.utils.tensorflow.constants import EPOCHS, RANDOM_SEED


# from tests.nlu.utilities import write_file_config

def component_builder():
    return ComponentBuilder()


def blank_config() -> RasaNLUModelConfig:
    return RasaNLUModelConfig({"language": "en", "pipeline": []})


def spacy_nlp_(component_builder, blank_config):
    spacy_nlp_config = {"name": "SpacyNLP"}
    return component_builder.create_component(spacy_nlp_config, blank_config).nlp

spacy_nlp = spacy_nlp_(component_builder(), blank_config())