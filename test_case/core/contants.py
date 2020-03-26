import asyncio
import os

from sanic.request import Request
import uuid
from datetime import datetime

from typing import Text, Iterator

import pytest
from _pytest.tmpdir import TempdirFactory

import rasa.utils.io
from rasa.core.agent import Agent
from rasa.core.channels.channel import CollectingOutputChannel, OutputChannel
from rasa.core.domain import Domain, SessionConfig
from rasa.core.events import ReminderScheduled, UserUttered, ActionExecuted
from rasa.core.interpreter import RegexInterpreter
from rasa.core.nlg import TemplatedNaturalLanguageGenerator
from rasa.core.policies.ensemble import PolicyEnsemble, SimplePolicyEnsemble
from rasa.core.policies.memoization import (
    AugmentedMemoizationPolicy,
    MemoizationPolicy,
    Policy,
)
from rasa.core.processor import MessageProcessor
from rasa.core.slots import Slot
from rasa.core.tracker_store import InMemoryTrackerStore, MongoTrackerStore
from rasa.core.trackers import DialogueStateTracker


DEFAULT_DOMAIN_PATH_WITH_SLOTS = "/Users/hao/Projects/Ftech/Onboardings/rasa-nlp/data/test_domains/default_with_slots.yml"
# DEFAULT_DOMAIN_PATH_WITH_SLOTS = "data/test_domains/default_with_slots.yml"

DEFAULT_DOMAIN_PATH_WITH_SLOTS_AND_NO_ACTIONS = (
    "data/test_domains/default_with_slots_and_no_actions.yml"
)

DEFAULT_DOMAIN_PATH_WITH_MAPPING = "data/test_domains/default_with_mapping.yml"

DEFAULT_STORIES_FILE = "/Users/hao/Projects/Ftech/Onboardings/rasa-nlp/data/test_stories/stories_defaultdomain.md"

DEFAULT_STACK_CONFIG = "data/test_config/stack_config.yml"

DEFAULT_NLU_DATA = "examples/moodbot/data/nlu.md"

END_TO_END_STORY_FILE = "data/test_evaluations/end_to_end_story.md"

E2E_STORY_FILE_UNKNOWN_ENTITY = "data/test_evaluations/story_unknown_entity.md"

MOODBOT_MODEL_PATH = "examples/moodbot/models/"

RESTAURANTBOT_PATH = "examples/restaurantbot/"

DEFAULT_ENDPOINTS_FILE = "data/test_endpoints/example_endpoints.yml"

TEST_DIALOGUES = [
    "data/test_dialogues/default.json",
    "data/test_dialogues/formbot.json",
    "data/test_dialogues/moodbot.json",
    "data/test_dialogues/restaurantbot.json",
]

EXAMPLE_DOMAINS = [
    DEFAULT_DOMAIN_PATH_WITH_SLOTS,
    DEFAULT_DOMAIN_PATH_WITH_SLOTS_AND_NO_ACTIONS,
    DEFAULT_DOMAIN_PATH_WITH_MAPPING,
    "examples/formbot/domain.yml",
    "examples/moodbot/domain.yml",
    "examples/restaurantbot/domain.yml",
]
templates = {
        "utter_ask_rephrase": [{"text": "can you rephrase that?"}],
        "utter_restart": [{"text": "congrats, you've restarted me!"}],
        "utter_back": [{"text": "backing up..."}],
        "utter_invalid": [{"text": "a template referencing an invalid {variable}."}],
        "utter_buttons": [
            {
                "text": "button message",
                "buttons": [
                    {"payload": "button1", "title": "button1"},
                    {"payload": "button2", "title": "button2"},
                ],
            }
        ],
    }
template_nlg = TemplatedNaturalLanguageGenerator(templates)
default_channel = CollectingOutputChannel()
default_domain = Domain.load(DEFAULT_DOMAIN_PATH_WITH_SLOTS)
default_nlg = TemplatedNaturalLanguageGenerator(default_domain.templates)
default_tracker = DialogueStateTracker("my-sender", default_domain.slots)
template_sender_tracker = DialogueStateTracker("template-sender", default_domain.slots)