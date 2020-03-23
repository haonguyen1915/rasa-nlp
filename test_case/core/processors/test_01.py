import asyncio

import datetime
import pytest
import time
import uuid
import json
from _pytest.monkeypatch import MonkeyPatch
from typing import Optional, Text, List
from unittest.mock import patch

from rasa.core import jobs
from rasa.core.actions.action import ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME

from rasa.core.agent import Agent
from rasa.core.channels.channel import CollectingOutputChannel, UserMessage
from rasa.core.domain import SessionConfig
from rasa.core.events import (
    ActionExecuted,
    BotUttered,
    ReminderCancelled,
    ReminderScheduled,
    Restarted,
    UserUttered,
    SessionStarted,
    Event,
    SlotSet,
)
from rasa.core.interpreter import RasaNLUHttpInterpreter
from rasa.core.processor import MessageProcessor, DEFAULT_INTENTS
from rasa.core.slots import Slot
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import EndpointConfig
from rasa.core.domain import Domain, SessionConfig
from rasa.core.constants import EXTERNAL_MESSAGE_PREFIX, IS_EXTERNAL
from rasa.core.tracker_store import InMemoryTrackerStore, MongoTrackerStore
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.utils.common import TempDirectoryPath
from rasa.core.policies.memoization import AugmentedMemoizationPolicy
import logging
from haolib import *
from constants_ import *

prj_dir = "{}/test_case/core/processors".format(PRJ_DIR)


async def _trained_default_agent() -> Tuple[Agent, str]:
    model_path = "{}/models/".format(prj_dir)

    agent = Agent(
        "{}/data/default_with_slots.yml".format(prj_dir),
        policies=[AugmentedMemoizationPolicy(max_history=3)],
    )

    training_data = await agent.load_data("{}/data/stories_defaultdomain.md".format(prj_dir))
    agent.train(training_data)
    agent.persist(model_path)
    return agent


def reset_conversation_state(agent) -> Agent:
    # Clean tracker store after each test so tests don't affect each other
    agent.tracker_store = InMemoryTrackerStore(agent.domain)
    agent.domain.session_config = SessionConfig.default()
    return agent


async def default_agent() -> Agent:
    trained_default_agent = await _trained_default_agent()
    return reset_conversation_state(trained_default_agent)


def default_domain():
    return Domain.load("{}/data/default_with_slots.yml".format(prj_dir).format(prj_dir))


async def default_processor() -> MessageProcessor:
    default_agent_ = await default_agent()
    tracker_store = InMemoryTrackerStore(default_agent_.domain)
    return MessageProcessor(
        default_agent_.interpreter,
        default_agent_.policy_ensemble,
        default_agent_.domain,
        tracker_store,
        TemplatedNaturalLanguageGenerator(default_agent_.domain.templates),
    )


def default_channel():
    return CollectingOutputChannel()


async def test_message_processor():
    default_channel_ = default_channel()
    default_processor_ = await default_processor()
    await default_processor_.handle_message(
        UserMessage('/greet{"name":"Core"}', default_channel_)
    )
    assert default_channel_.latest_output() == {
        "recipient_id": "default",
        "text": "hey there Core!",
    }


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_message_processor())
