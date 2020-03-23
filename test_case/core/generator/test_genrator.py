import asyncio
from rasa.core import training
from rasa.core.domain import Domain
from rasa.core.policies.memoization import AugmentedMemoizationPolicy, MemoizationPolicy
from test_case.core.contants import *
from haolib import *
from constants_ import *
prj_dir = "{}/test_case/core/generator".format(PRJ_DIR)
async def test_generator():
    default_domain = Domain.load("{}/domain_with_slots.yml".format(prj_dir))
    stories_file = "{}/data/stories.md".format(prj_dir)
    trackers = await training.load_data(
        stories_file, default_domain, augmentation_factor=0, debug_plots=True
    )
    # viz_trackers(trackers)
    print(trackers[0].get_states())
    print(trackers[1].get_states())
    print(trackers[2].get_states())
    print(trackers[3].get_states())
#
if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_generator())
