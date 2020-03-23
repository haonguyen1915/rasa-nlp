# from rasa_nlu.converters import load_data
from rasa_nlu.training_data import load_data
from rasa_nlu.config import RasaNLUModelConfig
from rasa_nlu.model import Trainer
from rasa_nlu.model import Metadata, Interpreter
from haolib import *
from constants_ import *

prj_dir = "{}/example/02_Weatherbot/".format(PRJ_DIR)
print(prj_dir)


# exit()

def train_nlu(data, config, model_dir):
    training_data = load_data(data)
    rasa_model_config = RasaNLUModelConfig(load_json(config))
    trainer = Trainer(rasa_model_config)
    trainer.train(training_data)
    model_directory = trainer.persist(model_dir, fixed_model_name='weathernlu')


def run_nlu():
    # interpreter = Interpreter.load('{}/models/nlu/default/weathernlu'.format(prj_dir),
    #                                RasaNLUModelConfig(load_json('{}/config_spacy.json'.format(prj_dir))))
    interpreter = Interpreter.load('{}/models/nlu/default/weathernlu'.format(prj_dir))
    show_dict(interpreter.parse("I am planning my holiday to Lithuania. I wonder what is the weather out there."))


if __name__ == '__main__':
    # train_nlu('{}/data/data.json'.format(prj_dir), '{}/config_spacy.json'.format(prj_dir), '{}/models/nlu'.format(prj_dir))
    run_nlu()
