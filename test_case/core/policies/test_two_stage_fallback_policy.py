from constants_ import *
from rasa.core.actions.action import (
    ACTION_DEFAULT_ASK_AFFIRMATION_NAME,
    ACTION_DEFAULT_ASK_REPHRASE_NAME,
    ACTION_DEFAULT_FALLBACK_NAME,
    ACTION_LISTEN_NAME,
    ActionRevertFallbackEvents,
    ACTION_RESTART_NAME,
    ACTION_BACK_NAME,
)
from rasa.core.policies.two_stage_fallback import TwoStageFallbackPolicy
from test_case.core.utils import *
from test_case.core.contants import *

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
    # viz_tracker(tracker)
    action = ActionRevertFallbackEvents()
    events += await action.run(channel, nlg, tracker, domain)
    # events = await action.run(channel, nlg, tracker, domain)
    # viz_events(events)
    # exit()
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
    # tracker = DialogueStateTracker.from_events("🕵️‍♀️", events)
    # appied_events = tracker.applied_events()
    # viz_events(appied_events)

    tracker = await _get_tracker_after_reverts(
        events, default_channel, default_nlg, default_domain
    )
    # viz_tracker(tracker)
    # print(tracker.export_stories())

    assert "greet" == tracker.latest_message.parse_data["intent"]["name"]
    assert tracker.export_stories() == (
        "## sender\n* greet\n    - utter_hello\n* greet\n"
    )
    # tracker = DialogueStateTracker.from_events("🕵️‍♀️", events)
    # viz_events(tracker.events)

    appied_events = tracker.applied_events()
    # viz_events(appied_events)
    # print(tracker.export_stories())


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


async def test_successful_rephrasing():
    events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet", 0.2),
        ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("deny", 1),
        ActionExecuted(ACTION_DEFAULT_ASK_REPHRASE_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("bye", 1),
    ]

    tracker = await _get_tracker_after_reverts(
        events, default_channel, default_nlg, default_domain
    )
    # viz_events(tracker.events)

    assert "bye" == tracker.latest_message.parse_data["intent"]["name"]
    assert tracker.export_stories() == "## sender\n* bye\n"

    # viz_events(tracker.applied_events())


def test_affirm_rephrased_intent():
    trained_policy = TwoStageFallbackPolicy.load("{}/models/two_stage_fallback".format(prj_dir))
    events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet", 0.2),
        ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("deny", 1),
        ActionExecuted(ACTION_DEFAULT_ASK_REPHRASE_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet", 0.2),
    ]

    next_action = _get_next_action(trained_policy, events, default_domain)

    assert next_action == ACTION_DEFAULT_ASK_AFFIRMATION_NAME


async def test_affirmed_rephrasing():
    events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet", 0.2),
        ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("deny", 1),
        ActionExecuted(ACTION_DEFAULT_ASK_REPHRASE_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("bye", 0.2),
        ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("bye", 1),
    ]

    tracker = await _get_tracker_after_reverts(
        events, default_channel, default_nlg, default_domain
    )
    # viz_events(tracker.events)
    assert "bye" == tracker.latest_message.parse_data["intent"]["name"]
    assert tracker.export_stories() == "## sender\n* bye\n"
    # viz_events(tracker.applied_events())


def test_denied_rephrasing_affirmation():
    trained_policy = TwoStageFallbackPolicy.load("{}/models/two_stage_fallback".format(prj_dir))
    events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet", 0.2),
        ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("deny", 1),
        ActionExecuted(ACTION_DEFAULT_ASK_REPHRASE_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("bye", 0.2),
        ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("deny", 1),
    ]

    next_action = _get_next_action(trained_policy, events, default_domain)

    assert next_action == ACTION_DEFAULT_FALLBACK_NAME


async def test_rephrasing_instead_affirmation():
    events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet", 1),
        ActionExecuted("utter_hello"),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet", 0.2),
        ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("bye", 1),
    ]

    tracker = await _get_tracker_after_reverts(
        events, default_channel, default_nlg, default_domain
    )

    assert "bye" == tracker.latest_message.parse_data["intent"]["name"]
    assert tracker.export_stories() == (
        "## sender\n* greet\n    - utter_hello\n* bye\n"
    )
    viz_events(tracker.events)
    viz_events(tracker.applied_events())


def test_unknown_instead_affirmation():
    trained_policy = TwoStageFallbackPolicy.load("{}/models/two_stage_fallback".format(prj_dir))
    events = [
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet", 0.2),
        ActionExecuted(ACTION_DEFAULT_ASK_AFFIRMATION_NAME),
        ActionExecuted(ACTION_LISTEN_NAME),
        user_uttered("greet", 0.2),
    ]

    next_action = _get_next_action(trained_policy, events, default_domain)

    assert next_action == ACTION_DEFAULT_FALLBACK_NAME


def test_listen_after_hand_off():
    trained_policy = TwoStageFallbackPolicy.load("{}/models/two_stage_fallback".format(prj_dir))
    events = [ActionExecuted(ACTION_DEFAULT_FALLBACK_NAME)]
    next_action = _get_next_action(trained_policy, events, default_domain)
    assert next_action == ACTION_LISTEN_NAME


if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(test_train_fallback_policy())
    # test_ask_affirmation()
    # loop.run_until_complete(test_affirmation())
    # test_ask_rephrase()
    # loop.run_until_complete(test_successful_rephrasing())
    # test_affirm_rephrased_intent()
    # loop.run_until_complete(test_affirmed_rephrasing())
    # test_denied_rephrasing_affirmation()
    # loop.run_until_complete(test_rephrasing_instead_affirmation())
    test_unknown_instead_affirmation()
    test_listen_after_hand_off()
