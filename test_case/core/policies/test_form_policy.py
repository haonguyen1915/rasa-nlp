import asyncio
from constants_ import *
from rasa.core import training
from rasa.core.domain import Domain
from rasa.core.policies.form_policy import FormPolicy

prj_dir = "{}/test_case/core/policies".format(PRJ_DIR)

domain = Domain.load("{}/data/form.yml".format(prj_dir))


async def test_train_form_policy():
    policy = FormPolicy()
    trackers = await training.load_data("{}/data/stories_form.md".format(prj_dir),
                                        domain,
                                        augmentation_factor=0)
    policy.train(trackers, domain)
    policy.persist("{}/models/form".format(prj_dir))


async def test_infer_form_policy():
    policy = FormPolicy.load("{}/models/form".format(prj_dir))

    trackers = await training.load_data("{}/data/stories_form.md".format(prj_dir),
                                        domain,
                                        augmentation_factor=0)
    all_states, all_actions = policy.featurizer.training_states_and_actions(trackers, domain)

    for tracker, states, actions in zip(trackers, all_states, all_actions):
        for state in states:
            if state is not None:
                # check that 'form: inform' was ignored
                assert "intent_inform" not in state.keys()
        recalled = policy.recall(states, tracker, domain)
        active_form = policy._get_active_form_name(states[-1])
        print(states)
        print(recalled)
        print(active_form)

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    # loop.run_until_complete(test_train_form_policy())
    loop.run_until_complete(test_infer_form_policy())
