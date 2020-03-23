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
from rasa.core.policies.memoization import AugmentedMemoizationPolicy, MemoizationPolicy
from test_case.core.contants import *
from haolib import *
from constants_ import *
from texttable import Texttable

prj_dir = "{}/test_case/core/generator".format(PRJ_DIR)


def print_data_training(decoded, actions):
    t = Texttable(max_width=130)
    t.add_rows([["Decoded", "Action"]])
    for decode, action in zip(decoded, actions):
        t.add_row([str(decode), str(action)])
    print(t.draw())


def print_data_training(decoded, actions):
    t = Texttable(max_width=130)
    t.add_rows([["Decoded", "Action"]])
    for decode, action in zip(decoded, actions):
        t.add_row([str(decode), str(action)])
    print(t.draw())


async def test_FullDialogueTrackerFeaturizer():
    # viz_domain(default_domain)
    default_domain = Domain.load("{}/domain_with_slots.yml".format(prj_dir))
    stories_file = "{}/data/stories.md".format(prj_dir)

    trackers = await training.load_data(
        stories_file, default_domain, augmentation_factor=0, debug_plots=True
    )
    # viz_trackers(trackers)
    featurizer = FullDialogueTrackerFeaturizer(state_featurizer=BinarySingleStateFeaturizer())
    (trackers_as_states, trackers_as_actions) = featurizer.training_states_and_actions(trackers, default_domain)
    # print_data_training(trackers_as_states, trackers_as_actions)
    # show_dict(trackers_as_states.log)
    # Featurize
    # dialog_training_data = featurizer.featurize_trackers(trackers, default_domain)
    # print(dialog_training_data.X)


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_FullDialogueTrackerFeaturizer())
