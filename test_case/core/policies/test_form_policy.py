import asyncio
from rasa.core import training
from rasa.core.domain import Domain
from rasa.core.policies.mapping_policy import MappingPolicy
from rasa.core.channels.channel import CollectingOutputChannel, UserMessage
from test_case.core.contants import *
from haolib import *
from haolib.lib_rasa.helpers import *
from constants_ import *
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
from rasa.core.events import ActionExecuted, UserUttered
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
import typing
from typing import Text, List, Optional, Text, Any, Dict

prj_dir = "{}/test_case/core/policies".format(PRJ_DIR)


def user_uttered(
        text: Text, confidence: float, metadata: Dict[Text, Any] = None
) -> UserUttered:
    parse_data = {"intent": {"name": text, "confidence": confidence}}
    return UserUttered(
        text="Random",
        intent=parse_data["intent"],
        parse_data=parse_data,
        metadata=metadata,
    )

intent_mapping = [
    ("default", "utter_default"),
    ("greet", "utter_greet"),
    (USER_INTENT_RESTART, ACTION_RESTART_NAME),
    (USER_INTENT_BACK, ACTION_BACK_NAME),
]


async def test_maping_policy():
    domain = Domain.load("{}/data/form.yml".format(prj_dir))
    policy = FormPolicy()
    events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered(intent_mapping[0][0], 1),
    ]
    trackers = await training.load_data("data/test_stories/stories_form.md", domain)
    policy.train(trackers, domain)

    (
        all_states,
        all_actions,
    ) = policy.featurizer.training_states_and_actions(trackers, domain)


    tracker = DialogueStateTracker.from_events("sender", events, [], 20)
    scores = policy.predict_action_probabilities(tracker, default_domain)
    index = scores.index(max(scores))
    print("action names: {}".format(default_domain.action_names))
    print(default_domain.action_names[index])

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_maping_policy())
