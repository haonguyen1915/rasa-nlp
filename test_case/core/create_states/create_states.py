import json
import asyncio
from haolib import *
from rasa.core import training, utils
from rasa.core.domain import Domain, InvalidDomain, SessionConfig
from rasa.core.featurizers import FullDialogueTrackerFeaturizer, BinarySingleStateFeaturizer
from test_case.core.contants import *
import asyncio
from rasa.core import training
from rasa.core.domain import Domain
from constants_ import *

prj_dir = "{}/test_case/core/create_states".format(PRJ_DIR)

def print_data_training(decoded, actions):
    t = Texttable(max_width=130)
    t.add_rows([["Decoded", "Action"]])
    for decode, action in zip(decoded, actions):
        t.add_row([str(decode), str(action)])
    print(t.draw())

async def test_FullDialogueTrackerFeaturizer():
    # viz_domain(default_domain)
    default_domain = Domain.load("{}/data/domain_with_slots.yml".format(prj_dir))
    stories_file = "{}/data/stories.md".format(prj_dir)

    trackers = await training.load_data(
        stories_file, default_domain, augmentation_factor=0, debug_plots=False
    )
    # viz_trackers(trackers)
    featurizer = FullDialogueTrackerFeaturizer(state_featurizer=BinarySingleStateFeaturizer())
    # print_title("START TRAINING STATES")
    (trackers_as_states, trackers_as_actions) = featurizer.training_states_and_actions(trackers, default_domain)
    print_data_training(trackers_as_states, trackers_as_actions)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_FullDialogueTrackerFeaturizer())
