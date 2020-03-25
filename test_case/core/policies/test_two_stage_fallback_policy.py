import asyncio
from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy
from constants_ import *
from rasa.core.domain import Domain
from unittest.mock import Mock, patch

import numpy as np
import pytest

from rasa.core import training
from rasa.core.actions.action import (
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_LISTEN_NAME,
    ActionRevertFallbackEvents,
    ACTION_RESTART_NAME,
    ACTION_BACK_NAME,
)
from rasa.core.constants import USER_INTENT_RESTART, USER_INTENT_BACK
from rasa.core.channels.channel import UserMessage
from rasa.core.domain import Domain
from rasa.core.events import ActionExecuted, ConversationPaused
from rasa.core.featurizers import (
    BinarySingleStateFeaturizer,
    LabelTokenizerSingleStateFeaturizer,
    MaxHistoryTrackerFeaturizer,
    FullDialogueTrackerFeaturizer,
)
from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy
from rasa.core.policies.ted_policy import TEDPolicy
from rasa.core.policies.fallback import FallbackPolicy
from rasa.core.policies.form_policy import FormPolicy
from rasa.core.policies.keras_policy import KerasPolicy
from rasa.core.policies.mapping_policy import MappingPolicy
from rasa.core.policies.memoization import AugmentedMemoizationPolicy, MemoizationPolicy
from rasa.core.policies.sklearn_policy import SklearnPolicy
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.tensorflow.constants import (
    SIMILARITY_TYPE,
    RANKING_LENGTH,
    LOSS_TYPE,
    SCALE_LOSS,
    EVAL_NUM_EXAMPLES,
    EPOCHS,
    KEY_RELATIVE_ATTENTION,
    VALUE_RELATIVE_ATTENTION,
    MAX_RELATIVE_POSITION,
)
from rasa.utils import train_utils
from rasa.core.channels.channel import CollectingOutputChannel, OutputChannel
from test_case.core.utils import *
from test_case.core.contants import *

# from tests.core.conftest import (
#     DEFAULT_DOMAIN_PATH_WITH_MAPPING,
#     DEFAULT_DOMAIN_PATH_WITH_SLOTS,
#     DEFAULT_STORIES_FILE,
# )
# from tests.core.utilities import get_tracker, read_dialogue_file, user_uttered
#

# async def train_trackers(domain, augmentation_factor=20):
#     return await training.load_data(
#         DEFAULT_STORIES_FILE, domain, augmentation_factor=augmentation_factor
#     )


prj_dir = "{}/test_case/core/policies".format(PRJ_DIR)
content = """
        intents:
          - greet
          - bye
          - affirm
          - deny
        """
default_domain = Domain.from_yaml(content)
default_channel = CollectingOutputChannel()


def _get_next_action(policy, events, domain):
    tracker = get_tracker(events)

    scores = policy.predict_action_probabilities(tracker, domain)
    index = scores.index(max(scores))
    return domain.action_names[index]


async def _get_tracker_after_reverts(events, channel, nlg, domain):
    tracker = get_tracker(events)
    action = ActionRevertFallbackEvents()
    events += await action.run(channel, nlg, tracker, domain)

    return get_tracker(events)


async def test_train_fallback_policy():
    policy = TwoStageFallbackPolicy(priority=1, deny_suggestion_intent_name="deny")
    policy.persist("{}/models/two_stage_fallback".format(prj_dir))


def test_ask_affirmation():
    policy = TwoStageFallbackPolicy.load("{}/models/two_stage_fallback".format(prj_dir))
    events = [ActionExecuted(ACTION_LISTEN_NAME), user_uttered("Hi", 0.2)]

    next_action = _get_next_action(policy, events, default_domain)

    assert next_action == ACTION_DEFAULT_ASK_AFFIRMATION_NAME


async def test_affirmation():
    events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet", 1),
        ActionExecuted("utter_hello"),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet", 0.2),
        ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet", 1),
    ]

    tracker = await _get_tracker_after_reverts(
        events, default_channel, default_nlg, default_domain
    )
    # viz_tracker(tracker)
    # print(tracker.export_stories())

    assert "greet" == tracker.latest_message.parse_data["intent"]["name"]
    assert tracker.export_stories() == (
        "## sender\n* greet\n    - utter_hello\n* greet\n"
    )
def test_ask_rephrase():
    policy = TwoStageFallbackPolicy.load("{}/models/two_stage_fallback".format(prj_dir))
    events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet", 0.2),
        ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("deny", 1),
    ]

    next_action = _get_next_action(policy, events, default_domain)

    assert next_action == ACTION_DEFAULT_ASK_REPHRASE_NAME

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(test_train_fallback_policy())
    # loop.run_until_complete(test_ask_affirmation())
    test_ask_affirmation()
    loop.run_until_complete(test_affirmation())
    test_ask_rephrase()
