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
prj_dir = get_cfd(backward=2)
DEFAULT_DATA_PATH = "/Users/hao/Projects/Ftech/Library/rasa/data/examples/rasa/demo-rasa.json"
model_dir = "{}/models".format(file_dỉr)

def test_train_model_without_data():
    td = load_data(DEFAULT_DATA_PATH)
    # language, pipeline = pipelines_for_tests()[1]
    # show_dict(pipeline)
    # exit()
    language = "en"
    pipeline = load_json("{}/test_case/test_pipelines/config_pipeline.json".format(prj_dir))
    # exit()
    _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})

    trainer = Trainer(_config)
    trainer.train(td)
    persisted_path = trainer.persist(model_dir)
    loaded = Interpreter.load(persisted_path)
    assert loaded.pipeline

    # Inference
    # result = loaded.parse("i'm looking for a place in the north of town")
    result = loaded.parse("show me chinese restaurants")
    result = dict(filter(lambda item: item[0] not in ["intent_ranking"], result.items()))
    show_dict(result)


def reference():
    # language, pipeline = pipelines_for_tests()[1]
    # _config = RasaNLUModelConfig({"pipeline": pipeline, "language": language})
    loaded = Interpreter.load(
        "/Users/hao/Projects/Library/basic-python-pytorch/test_case/test_pipelines/models/nlu_20200306-174224")
    show_interpreter(loaded)
    assert isinstance(loaded, Interpreter)
    # Inference
    # result2 = loaded.parse("i'm looking for a place in the north of town")
    # result2 = loaded.parse("show me chinese restaurants")
    result2 = loaded.parse("central indian restaurant")
    # result2 = loaded.parse("the cost is 50$ and the time is 2017-04-10T00:00:00.000+02:00")
    # show_dict(result1)
    result2 = dict(filter(lambda item: item[0] not in ["intent_ranking"], result2.items()))
    show_dict(result2)


if __name__ == "__main__":
    # test_train_model_without_data()
    reference()
