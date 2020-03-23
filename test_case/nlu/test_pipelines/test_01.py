import os
import pytest

from rasa.nlu import registry, train
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter, Trainer, Metadata
from rasa.nlu.training_data import TrainingData, load_data
from rasa.utils.tensorflow.constants import EPOCHS
from haolib.lib_rasa.utils import *
from haolib import *

file_dỉr = get_cfd()
DEFAULT_DATA_PATH = "/Users/hao/Projects/Ftech/Library/rasa/data/examples/rasa/demo-rasa.json"
model_dir = "{}/models".format(file_dỉr)


def as_pipeline(*components):
    return [{"name": c, EPOCHS: 1} for c in components]


def pipelines_for_tests():
    # these templates really are just for testing
    # every component should be in here so train-persist-load-use cycle can be
    # tested they still need to be in a useful order - hence we can not simply
    # generate this automatically.

    # Create separate test pipelines for dense featurizers
    # because they can't co-exist in the same pipeline together,
    # as their tokenizers_todo break the incoming message into different number of tokens.

    # first is language followed by list of components
    return [
        ("en", as_pipeline("KeywordIntentClassifier")),
        (
            "en",
            as_pipeline(
                "WhitespaceTokenizer",
                "RegexFeaturizer",
                "LexicalSyntacticFeaturizer",
                "CountVectorsFeaturizer",
                "CRFEntityExtractor",
                "DucklingHTTPExtractor",
                "DIETClassifier",
                "EmbeddingIntentClassifier",
                "ResponseSelector",
                "EntitySynonymMapper",
            ),
        ),
        (
            "en",
            as_pipeline(
                "SpacyNLP",
                "SpacyTokenizer",
                "SpacyFeaturizer",
                "SpacyEntityExtractor",
                "SklearnIntentClassifier",
            ),
        ),
        (
            "en",
            as_pipeline(
                "HFTransformersNLP",
                "LanguageModelTokenizer",
                "LanguageModelFeaturizer",
                "DIETClassifier",
            ),
        ),
        ("en", as_pipeline("ConveRTTokenizer", "ConveRTFeaturizer", "DIETClassifier")),
        (
            "en",
            as_pipeline(
                "MitieNLP", "MitieTokenizer", "MitieFeaturizer", "MitieIntentClassifier"
            ),
        ),
        (
            "zh",
            as_pipeline(
                "MitieNLP", "JiebaTokenizer", "MitieFeaturizer", "MitieEntityExtractor"
            ),
        ),
    ]


def test_train_model_without_data():
    td = load_data(DEFAULT_DATA_PATH)
    language, pipeline = pipelines_for_tests()[1]
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})
    show_dict({"pipeline": pipeline, "language": language})
    exit()

    trainer = Trainer(_config)
    trainer.train(td)
    persisted_path = trainer.persist(model_dir)
    loaded = Interpreter.load(persisted_path)
    assert loaded.pipeline

    # Inference
    result = loaded.parse("i'm looking for a place in the north of town")
    result = loaded.parse("show me chinese restaurants")
    result = dict(filter(lambda item: item[0] not in ["intent_ranking"], result.items()))
    show_dict(result)


def reference():
    language, pipeline = pipelines_for_tests()[1]
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})
    loaded = Interpreter.load(
        "/Users/hao/Projects/Library/basic-python-pytorch/test_case/test_pipelines/models/test_01_models_weights")

    print(type(loaded))
    show_interpreter(loaded)
    assert isinstance(loaded, Interpreter)
    # Inference
    result2 = loaded.parse("i'm looking for a place in the north of town")
    # result2 = loaded.parse("show me chinese restaurants")
    # result2 = loaded.parse("central indian restaurant")
    # result2 = loaded.parse("the cost is 50$ and the time is 2017-04-10T00:00:00.000+02:00")
    # show_dict(result1)
    result2 = dict(filter(lambda item: item[0] not in ["intent_ranking"], result2.items()))
    show_dict(result2)


if __name__ == "__main__":
    test_train_model_without_data()
    # reference()
