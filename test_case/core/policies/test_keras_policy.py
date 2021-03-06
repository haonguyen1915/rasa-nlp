import asyncio
from rasa.core import training
from rasa.core.domain import Domain
from rasa.core.policies.memoization import AugmentedMemoizationPolicy, MemoizationPolicy
from rasa.core.channels.channel import CollectingOutputChannel, UserMessage
from test_case.core.contants import *
from haolib import *
from haolib.lib_rasa.helpers import *
from rasa.core.featurizers import (
    BinarySingleStateFeaturizer,
    LabelTokenizerSingleStateFeaturizer,
    MaxHistoryTrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
)
from rasa.core.policies.keras_policy import KerasPolicy
from constants_ import *

prj_dir = "{}/test_case/core/policies".format(PRJ_DIR)

def featurizer():
    featurizer = MaxHistoryTrackerFeaturizer(
        BinarySingleStateFeaturizer(), max_history=3
    )
    return featurizer

async def test_train_keras_policy():
    default_domain = Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS)
    trackers = await training.load_data(
        DEFAULT_STORIES_FILE, default_domain, augmentation_factor=0, debug_plots=False
    )
    policy = KerasPolicy(featurizer=featurizer(), priority=1)
    policy.train(trackers, default_domain)
    policy.persist("{}/models/keras".format(prj_dir))
#
#
async def test_infer():
    default_domain = Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS)
    print("action names: {}".format(default_domain.action_names))

    trackers = await training.load_data(
        DEFAULT_STORIES_FILE, default_domain, augmentation_factor=4, debug_plots=True
    )
    policy = KerasPolicy.load("{}/models/keras".format(prj_dir))
    for tracker in trackers:
        y_pred = policy.predict_action_probabilities(tracker, default_domain)
        index = y_pred.index(max(y_pred))
        print(default_domain.action_names[index])
        # print(default_domain.action_names[recalled])


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_train_keras_policy())
    # loop.run_until_complete(test_infer())
