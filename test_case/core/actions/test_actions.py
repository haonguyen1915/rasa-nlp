from typing import List

import pytest
import asyncio
import rasa.core
from rasa.core.actions import action
from rasa.core.actions.action import (
    ACTION_BACK_NAME,
    ACTION_DEACTIVATE_FORM_NAME,
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_LISTEN_NAME,
    ACTION_RESTART_NAME,
    ACTION_REVERT_FALLBACK_EVENTS_NAME,
    ACTION_SESSION_START_NAME,
    ActionBack,
    ActionDefaultAskAffirmation,
    ActionDefaultAskRephrase,
    ActionDefaultFallback,
    ActionExecutionRejection,
    ActionListen,
    ActionRestart,
    ActionUtterTemplate,
    ActionRetrieveResponse,
    RemoteAction,
    ActionSessionStart,
)
from rasa.core.channels import CollectingOutputChannel
from rasa.core.domain import Domain, SessionConfig
from rasa.core.events import (
    Restarted,
    SlotSet,
    UserUtteranceReverted,
    BotUttered,
    Form,
    SessionStarted,
    ActionExecuted,
    Event,
    UserUttered,
)
from rasa.core.nlg.template import TemplatedNaturalLanguageGenerator
from rasa.core.constants import USER_INTENT_SESSION_START
from rasa.core.trackers import DialogueStateTracker
from rasa.utils.endpoints import ClientResponseError, EndpointConfig
from test_case.core.contants import *
from haolib import *


async def test_action_restart():
    events = await ActionRestart().run(
        default_channel, template_nlg, template_sender_tracker, default_domain
    )

    assert events == [BotUttered("congrats, you've restarted me!"), Restarted()]

    tracker = DialogueStateTracker.from_events("🕵️‍♀️", events)
    applied_events = tracker.applied_events()
    # assert applied_events == [ActionExecuted(action_name=ACTION_LISTEN_NAME)]
    # viz_events(applied_events)


async def test_action_back():
    events = await ActionBack().run(
        default_channel, template_nlg, template_sender_tracker, default_domain
    )

    assert events == [
        BotUttered("backing up..."),
        UserUtteranceReverted(),
        UserUtteranceReverted(),
    ]
    tracker = DialogueStateTracker.from_events("🕵️‍♀️", events)
    applied_events = tracker.applied_events()
    # assert applied_events == [ActionExecuted(action_name=ACTION_LISTEN_NAME)]
    # viz_events(applied_events)


async def test_action_session_start_without_slots():
    events = await ActionSessionStart().run(
        default_channel, template_nlg, template_sender_tracker, default_domain
    )
    assert events == [SessionStarted(), ActionExecuted(ACTION_LISTEN_NAME)]


async def test_action_session_start_with_slots():
    # "session_config, expected_events",
    data_all = [
        (
            SessionConfig(123, True),
            [
                SessionStarted(),
                SlotSet("my_slot", "value"),
                SlotSet("another-slot", "value2"),
                ActionExecuted(action_name=ACTION_LISTEN_NAME),
            ],
        ),
        (
            SessionConfig(123, False),
            [SessionStarted(), ActionExecuted(action_name=ACTION_LISTEN_NAME)],
        ),
    ]
    data = data_all[1]
    # set a few slots on tracker
    slot_set_event_1 = SlotSet("my_slot", "value")
    slot_set_event_2 = SlotSet("another-slot", "value2")
    for event in [slot_set_event_1, slot_set_event_2]:
        template_sender_tracker.update(event)

    default_domain.session_config = data[0]
    # viz_tracker(template_sender_tracker)

    events = await ActionSessionStart().run(
        default_channel, template_nlg, template_sender_tracker, default_domain
    )
    # viz_events(events)
    # viz_tracker(template_sender_tracker)
    assert events == data[1]

    # make sure that the list of events has ascending timestamps
    assert sorted(events, key=lambda x: x.timestamp) == events

    tracker = DialogueStateTracker.from_events("🕵️‍♀️", events)
    applied_events = tracker.applied_events()
    assert applied_events == [ActionExecuted(action_name=ACTION_LISTEN_NAME)]
    # viz_events(applied_events)


async def test_applied_events_after_action_session_start():
    default_domain.session_config = SessionConfig(123, False)
    slot_set = SlotSet("my_slot", "value")
    events = [
        slot_set,
        ActionExecuted(ACTION_LISTEN_NAME),
        # User triggers a restart manually by triggering the intent
        UserUttered(
            text=f"/{USER_INTENT_SESSION_START}",
            intent={"name": USER_INTENT_SESSION_START},
        ),
    ]
    tracker = DialogueStateTracker.from_events("🕵️‍♀️", events)
    # viz_tracker(tracker)

    # Mapping Policy kicks in and runs the session restart action
    events = await ActionSessionStart().run(
        default_channel, template_nlg, tracker, Domain.empty()
    )
    # viz_events(events)
    for event in events:
        tracker.update(event)

    assert tracker.applied_events() == [slot_set, ActionExecuted(ACTION_LISTEN_NAME)]
    # viz_events(tracker.applied_events())


async def test_action_default_fallback():
    events = await ActionDefaultFallback().run(
        default_channel, default_nlg, default_tracker, default_domain
    )
    assert events == [
        BotUttered("sorry, I didn't get that, can you rephrase it?"),
        UserUtteranceReverted(),
    ]
    tracker = DialogueStateTracker.from_events("🕵️‍♀️", events)
    # viz_events(tracker.events)
    applied_events = tracker.applied_events()
    assert applied_events == []

    # viz_tracker(default_tracker)

    # viz_events(tracker)


async def test_action_default_ask_affirmation():
    events = await ActionDefaultAskAffirmation().run(
        default_channel, default_nlg, default_tracker, default_domain
    )
    assert events == [
        BotUttered(
            "Did you mean 'None'?",
            {
                "buttons": [
                    {"title": "Yes", "payload": "/None"},
                    {"title": "No", "payload": "/out_of_scope"},
                ]
            },
        )
    ]

    tracker = DialogueStateTracker.from_events("🕵️‍♀️", events)
    viz_events(tracker.events)
    applied_events = tracker.applied_events()
    viz_events(applied_events)

async def test_action_default_ask_rephrase():
    events = await ActionDefaultAskRephrase().run(
        default_channel, template_nlg, template_sender_tracker, default_domain
    )

    assert events == [BotUttered("can you rephrase that?")]

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_action_restart())
    loop.run_until_complete(test_action_back())
    # loop.run_until_complete(test_action_session_start_without_slots())
    loop.run_until_complete(test_action_session_start_with_slots())
    # loop.run_until_complete(test_applied_events_after_action_session_start())
    # loop.run_until_complete(test_action_default_fallback())
    # loop.run_until_complete(test_action_default_ask_affirmation())
    # loop.run_until_complete(test_action_default_ask_rephrase())
