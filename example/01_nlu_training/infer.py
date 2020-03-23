from rasa.nlu import registry, train
from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.model import Interpreter, Trainer, Metadata
from rasa.nlu.training_data import TrainingData, load_data
from rasa.utils.tensorflow.constants import EPOCHS
from haolib.lib_rasa.utils import *
from haolib import *
from constants_ import *
prj_dir = "{}/example/01_nlu_training".format(PRJ_DIR)


def reference():
    _config = RasaNLUModelConfig(load_json("{}/config.json".format(prj_dir)))
    loaded = Interpreter.load("{}/models/nlu_20200317-150056".format(prj_dir))

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
    reference()