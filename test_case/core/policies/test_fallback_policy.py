import asyncio
from rasa.core.policies.fallback import FallbackPolicy
from constants_ import *

prj_dir = "{}/test_case/core/policies".format(PRJ_DIR)


async def test_train_fallback_policy():
    policy = FallbackPolicy(priority=1)
    policy.persist("{}/models/fallback".format(prj_dir))


async def test_infer_fallback():
    policy = FallbackPolicy.load("{}/models/fallback".format(prj_dir))
    # "top_confidence, all_confidences, last_action_name, should_nlu_fallback",
    all_data = [
        (0.1, [0.1], "some_action", False),
        (0.1, [0.1], "action_listen", True),
        (0.9, [0.9, 0.1], "some_action", False),
        (0.9, [0.9, 0.1], "action_listen", False),
        (0.4, [0.4, 0.35], "some_action", False),
        (0.4, [0.4, 0.35], "action_listen", True),
        (0.9, [0.9, 0.85], "action_listen", True),
    ]
    for data in all_data:
        nlu_data = {
            "intent": {"confidence": data[0]},
            "intent_ranking": [
                {"confidence": confidence} for confidence in data[1]
            ],
        }
        should_nlu_fallback = policy.should_nlu_fallback(nlu_data, data[2])
        assert should_nlu_fallback == data[3]

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(test_train_fallback_policy())
    loop.run_until_complete(test_infer_fallback())
