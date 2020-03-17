import numpy as np
import pytest

import scipy.sparse

from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.tokenizers.whitespace_tokenizer import WhitespaceTokenizer
from rasa.nlu.featurizers.sparse_featurizer.lexical_syntactic_featurizer import (
    LexicalSyntacticFeaturizer,
)
from rasa.nlu.training_data import TrainingData
from rasa.nlu.constants import TEXT, SPARSE_FEATURE_NAMES, SPACY_DOCS
from rasa.nlu.training_data import Message
from haolib.lib_rasa.utils import *

samples = [
    (
        "HELLO goodbye hello",
        [
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 2.0, 2.0, 1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0, 1.0, 2.0],
        ],
    ),
    (
        "a 1",
        [
            [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
    ),
]


def test_text_featurizer():
    sentence, expected_features = samples[0]
    featurizer = LexicalSyntacticFeaturizer(
        {
            "features": [
                ["BOS", "upper"],
                ["BOS", "EOS", "prefix2", "digit"],
                ["EOS", "low"],
            ]
        }
    )

    train_message = Message(sentence)
    test_message = Message(sentence)

    WhitespaceTokenizer().process(train_message)
    WhitespaceTokenizer().process(test_message)
    show_message(train_message)
    featurizer.train(TrainingData([train_message]))
    show_message(train_message)

    featurizer.process(test_message)

    # assert isinstance(
    #     test_message.get(SPARSE_FEATURE_NAMES[TEXT]), scipy.sparse.coo_matrix
    # )
    #
    # actual = test_message.get(SPARSE_FEATURE_NAMES[TEXT]).toarray()
    #
    # assert np.all(actual == expected_features)

if __name__ == "__main__":
    test_text_featurizer()