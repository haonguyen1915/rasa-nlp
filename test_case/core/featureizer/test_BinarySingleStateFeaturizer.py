from rasa.core.featurizers import (
    TrackerFeaturizer,
    BinarySingleStateFeaturizer,
    LabelTokenizerSingleStateFeaturizer,
)
import numpy as np
from haolib import *


def test_BinarySingleStateFeaturizer():
    f = BinarySingleStateFeaturizer()
    f.input_state_map = {"a": 0, "b": 3, "c": 2, "d": 1}
    # "a"  "d"  "c"   "b"
    f.num_features = len(f.input_state_map)
    encoded = f.encode({"a": 1.0, "b": 1.0, "c": 0.0, "e": 1.0})
    assert is_numpy(encoded)
    assert list(encoded) == [1, 0, 0, 1]

    encoded = f.encode({"a": 1.0, "b": 0.1, "c": 0.2, "e": 1.0})
    assert is_numpy(encoded)
    assert list(encoded) == [1.0, 0.0, 0.2, 0.1]


if __name__ == "__main__":
    test_BinarySingleStateFeaturizer()
