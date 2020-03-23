from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter, Trainer, Metadata
from rasa.nlu.training_data import TrainingData, load_data
from rasa.utils.tensorflow.constants import EPOCHS
from haolib.lib_rasa.utils import *
from haolib import *
from constants_ import *

prj_dir = "{}/example/01_nlu_training".format(PRJ_DIR)


def train():
    td = load_data("{}/demo_rasa.json".format(prj_dir))
    _config = RasaNLUModelConfig(load_json("{}/config.json".format(prj_dir)))
    trainer = Trainer(_config)
    trainer.train(td)
    persisted_path = trainer.persist("{}/models".format(prj_dir))
    loaded = Interpreter.load(persisted_path)
    assert loaded.pipeline

    # Inference
    result = loaded.parse("i'm looking for a place in the north of town")
    result = loaded.parse("show me chinese restaurants")
    result = dict(filter(lambda item: item[0] not in ["intent_ranking"], result.items()))
    show_dict(result)


if __name__ == "__main__":
    train()
