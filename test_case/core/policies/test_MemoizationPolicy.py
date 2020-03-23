import asyncio
from rasa.core import training
from rasa.core.domain import Domain
from rasa.core.policies.memoization import AugmentedMemoizationPolicy, MemoizationPolicy
from rasa.core.channels.channel import CollectingOutputChannel, UserMessage
from test_case.core.contants import *
from haolib import *
from haolib.lib_rasa.helpers import *
from constants_ import *

prj_dir = "{}/test_case/core/policies".format(PRJ_DIR)


def print_data_training(decoded, actions):
    t = Texttable(max_width=130)
    t.add_rows([["Decoded", "Action"]])
    for decode, action in zip(decoded, actions):
        t.add_row([str(decode), str(action)])
    print(t.draw())


# def get_tracker(events) -> DialogueStateTracker:
#     return DialogueStateTracker.from_events("sender", events, [], 20)
#

# def train_trackers(domain, augmentation_factor=20):
#     return training.load_data(
#         DEFAULT_STORIES_FILE, domain, augmentation_factor=augmentation_factor
#     )


async def test_train_memorise():
    default_domain = Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS)
    trackers = await training.load_data(
        DEFAULT_STORIES_FILE, default_domain, augmentation_factor=50, debug_plots=False
    )
    policy = MemoizationPolicy(priority=1, max_history=None)
    policy.train(trackers, default_domain)
    policy.persist("{}/models/memorise".format(prj_dir))


async def test_infer():
    default_domain = Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS)
    trackers = await training.load_data(
        DEFAULT_STORIES_FILE, default_domain, augmentation_factor=0, debug_plots=True
    )
    policy = MemoizationPolicy.load("{}/models/memorise".format(prj_dir))
    for tracker in trackers:
        tracker_as_states = policy.featurizer.prediction_states([tracker], default_domain)
        states = tracker_as_states[0]
        print(states)
        states = [None, {}, {'prev_action_listen': 1.0, 'intent_greet': 1.0},
                  {'intent_greet': 1.0, 'prev_utter_greet': 1.0}, {'prev_action_listen': 1.0, 'intent_default': 1.0}]
        states = [{'prev_action_listen': 1.0, 'intent_greet': 1.0}, {'intent_greet': 1.0, 'prev_utter_greet': 1.0},
                  {'prev_action_listen': 1.0, 'intent_default': 1.0}, {'prev_utter_default': 1.0, 'intent_default': 1.0},
                  {'prev_action_listen': 1.0, 'intent_goodbye': 1.0}]
        recalled = policy.recall(states, tracker, default_domain)
        print(default_domain.action_names[recalled])


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(test_train_memorise())
    loop.run_until_complete(test_infer())
