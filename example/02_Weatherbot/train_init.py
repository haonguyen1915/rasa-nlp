from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import logging

from rasa_core.agent import Agent
from rasa_core.policies.keras_policy import KerasPolicy
from rasa_core.policies.memoization import MemoizationPolicy
from constants_ import *
from haolib import *
prj_dir = "{}/example/02_Weatherbot/".format(PRJ_DIR)
print(prj_dir)
if __name__ == '__main__':
    logging.basicConfig(level='INFO')

    training_data_file = './data/stories.md'.format(prj_dir)
    model_path = './models/dialogue'.format(prj_dir)

    agent = Agent('weather_domain.yml'.format(prj_dir), policies=[MemoizationPolicy(), KerasPolicy()])

    agent.train(
        training_data_file,
        augmentation_factor=50,
        max_history=2,
        epochs=500,
        batch_size=10,
        validation_split=0.2)

    agent.persist(model_path)
