import argparse
import asyncio
from constants_ import *
prj_dir = os.path.join(PRJ_DIR, "example/restaurantbot")
import sys
sys.path.append(prj_dir)
import logging
from typing import Text

import os
import rasa.utils.io
import rasa.train
from policy import RestaurantPolicy
from rasa.core.agent import Agent
from rasa.core.policies.memoization import MemoizationPolicy
from rasa.core.policies.mapping_policy import MappingPolicy
from haolib import *


logger = logging.getLogger(__name__)


# prj_dir = os.path.join(PR)

def train_nlu(config_file="config.yml", model_directory="models", model_name="current",
              training_data_file="data/nlu.md"):
    from rasa.nlu.training_data import load_data
    from rasa.nlu import config
    from rasa.nlu.model import Trainer

    training_data = load_data(training_data_file)
    trainer = Trainer(config.load(config_file))
    trainer.train(training_data)

    # Attention: trainer.persist stores the model and all meta data into a folder.
    # The folder itself is not zipped.
    model_path = os.path.join(model_directory, model_name)
    model_directory = trainer.persist(model_path, fixed_model_name="nlu")

    logger.info(f"Model trained. Stored in '{model_directory}'.")

    return model_directory


async def train_core(domain_file="domain.yml", model_directory="models", model_name="current",
                     training_data_file="data/stories.md"):
    agent = Agent(
        domain_file,
        policies=[
            MemoizationPolicy(max_history=3),
            MappingPolicy(),
            RestaurantPolicy(batch_size=100, epochs=100, validation_split=0.2),
        ],
    )
    training_data_file = "data/tiny_stories.md"
    training_data = await agent.load_data(training_data_file, augmentation_factor=10)
    # show_training_data(training_data)
    # print(type(training_data))
    # print(training_data)
    for data in training_data:
        print(type(data))
        # viz_domain(data.domain)
        # viz_TrackerWithCachedStates(data, view_domain=False)
    # exit()
    agent.train(training_data)

    # Attention: agent.persist stores the model and all meta data into a folder.
    # The folder itself is not zipped.
    model_path = os.path.join(model_directory, model_name, "core")
    agent.persist(model_path)

    logger.info(f"Model trained. Stored in '{model_path}'.")

    return model_path


async def parse(text: Text, model_path: Text):
    agent = Agent.load(model_path)

    response = await agent.handle_text(text)
    logger.info(f"Text: '{text}'")
    logger.info("Response:")
    logger.info(response)
    print(response)
    return response


async def conversation_loop(model_path: Text):
    agent = Agent.load(model_path)
    print("Press 'q' to exit the conversation")
    while True:
        val = input("input: ")
        if val == "q":
            break
        response = await agent.handle_text(val)
        print("Bot: {}".format(response[0]["text"]))
if __name__ == "__main__":
    loop = asyncio.get_event_loop()

    # train_nlu()
    # loop.run_until_complete(train_core())
    loop.run_until_complete(parse("hello", "models/current"))
    # loop.run_until_complete(conversation_loop("models/current"))
