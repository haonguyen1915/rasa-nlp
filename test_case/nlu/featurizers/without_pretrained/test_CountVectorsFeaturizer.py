import numpy as np
import pytest
import scipy.sparse

from rasa.nlu.config import RasaNLUModelConfig
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.constants import (
    CLS_TOKEN,
    TOKENS_NAMES,
    TEXT,
    INTENT,
    SPARSE_FEATURE_NAMES,
    RESPONSE,
)
from rasa.nlu.tokenizers.tokenizer import Token
from rasa.nlu.training_data import Message
from rasa.nlu.training_data import TrainingData
from rasa.nlu.featurizers.sparse_featurizer.count_vectors_featurizer import (
    CountVectorsFeaturizer,
)
from haolib.lib_rasa.utils import *
samples = [
    ("hello hello hello hello hello", [[1]], [[5]]),
    ("hello goodbye hello", [[0, 1]], [[1, 2]]),
    ("a b c d e f", [[1, 0, 0, 0, 0, 0]], [[1, 1, 1, 1, 1, 1]]),
    ("a 1 2", [[0, 1]], [[2, 1]]),
]


def test_count_vector_featurizer():
    sentence, expected, expected_cls = samples[1]
    ftr = CountVectorsFeaturizer({"token_pattern": r"(?u)\b\w+\b"})

    train_message = Message(sentence)
    test_message = Message(sentence)

    WhitespaceTokenizer().process(train_message)
    WhitespaceTokenizer().process(test_message)
    show_message(train_message)
    ftr.train(TrainingData([train_message]))
    show_message(train_message)
    ftr.process(test_message)

    assert isinstance(
        test_message.get(SPARSE_FEATURE_NAMES[TEXT]), scipy.sparse.coo_matrix
    )

    actual = test_message.get(SPARSE_FEATURE_NAMES[TEXT]).toarray()

    assert np.all(actual[0] == expected)
    assert np.all(actual[-1] == expected_cls)

if __name__ == "__main__":
    test_count_vector_featurizer()