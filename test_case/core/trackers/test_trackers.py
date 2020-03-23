import json
import logging
import os
import tempfile
from typing import List

import fakeredis
import pytest

import rasa.utils.io
from rasa.core import training, restore
from rasa.core.actions.action import ACTION_LISTEN_NAME, ACTION_SESSION_START_NAME
from rasa.core.agent import Agent
from rasa.core.domain import Domain
from rasa.core.events import (
    SlotSet,
    UserUttered,
    ActionExecuted,
    Restarted,
    ActionReverted,
    UserUtteranceReverted,
    SessionStarted,
    Event,
)
from rasa.core.tracker_store import (
    InMemoryTrackerStore,
    RedisTrackerStore,
    SQLTrackerStore,
)
from rasa.core.tracker_store import TrackerStore
from rasa.core.trackers import DialogueStateTracker, EventVerbosity
from test_case.core.trackers.utilities import *
# from tests.core.conftest import (
#     DEFAULT_STORIES_FILE,
#     EXAMPLE_DOMAINS,
#     TEST_DIALOGUES,
#     MockedMongoTrackerStore,
# )
# from tests.core.utilities import (
#     tracker_from_dialogue_file,
#     read_dialogue_file,
#     user_uttered,
#     get_tracker,
# )
from haolib import *
from constants_ import *

domain = Domain.load("data/moodbot.yml")


def test_tracker_duplicate():
    filename = "{}/data/test_dialogues/moodbot.json".format(PRJ_DIR)
    dialogue = read_dialogue_file(filename)
    tracker = DialogueStateTracker(dialogue.name, domain.slots)
    tracker.recreate_from_dialogue(dialogue)
    num_actions = len(
        [event for event in dialogue.events if isinstance(event, ActionExecuted)]
    )
    events = [event for event in dialogue.events if isinstance(event, ActionExecuted)]
    viz_events(dialogue.events)
    # print(type(events[0]).__name__)
    # exit()
    # viz_events(tracker.events)
    # viz_tracker(tracker, v_domain=True)
    # There is always one duplicated tracker more than we have actions,
    # as the tracker also gets duplicated for the
    # action that would be next (but isn't part of the operations)
    assert len(list(tracker.generate_all_prior_trackers())) == num_actions + 1

    # print(list(tracker.generate_all_prior_trackers()))
    viz_trackers(list(tracker.generate_all_prior_trackers()))


if __name__ == "__main__":
    test_tracker_duplicate()
