from rasa.core.featurizers import (
    TrackerFeaturizer,
    BinarySingleStateFeaturizer,
    LabelTokenizerSingleStateFeaturizer,
)
import numpy as np
from haolib import *


def test_LabelTokenizerSingleStateFeaturizer():
    f = LabelTokenizerSingleStateFeaturizer()
    f.user_labels = ["a_d"]
    f.bot_labels = ["c_b"]
    f.user_vocab = {"a": 0, "d": 1}
    f.bot_vocab = {"b": 1, "c": 0}
    f.num_features = len(f.user_vocab) + len(f.slot_labels) + len(f.bot_vocab)
    # Dictionary: 0    1    2    3
    #            "a"  "d"  "c"  "b"
    encoded = f.encode(
        {"a_d": 1.0, "prev_c_b": 0.0, "e": 1.0, "prev_action_listen": 1.0}
    )
    assert list(encoded) == [1, 1, 0, 0]
    # "a_d" -> ["a", "d"]  -> 1  1  0  0
    # "prev_c_b" -> ["prev", "c", "b"]  --> only "c_b" count: --> [0 0 0 0]
    # "prev_action_listen" --> ["prev", "action", "listen"] --> ignored because not in labels

    encoded = f.encode(
        {"a_d": 1.7, "prev_c_b": 2.0, "e": 1.0, "prev_action_listen": 1.0}
    )
    assert encoded == [1.7, 1.7, 2.0, 2.0]

if __name__ == "__main__":
    test_LabelTokenizerSingleStateFeaturizer()
