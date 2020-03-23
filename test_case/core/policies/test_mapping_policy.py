import asyncio
from constants_ import *
from rasa.core.actions.action import (
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_BACK_NAME,
)
from rasa.core.constants import USER_INTENT_RESTART, USER_INTENT_BACK
from rasa.core.domain import Domain
from rasa.core.events import ActionExecuted, UserUttered
from rasa.core.policies.mapping_policy import MappingPolicy
from rasa.core.trackers import DialogueStateTracker
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
    default_domain = Domain.load("{}/data/default_with_mapping.yml".format(prj_dir))
    policy = MappingPolicy()
    events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered(intent_mapping[0][0], 1),
    ]
    tracker = DialogueStateTracker.from_events("sender", events, [], 20)
    scores = policy.predict_action_probabilities(tracker, default_domain)
    index = scores.index(max(scores))
    print("action names: {}".format(default_domain.action_names))
    print(default_domain.action_names[index])


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_maping_policy())
