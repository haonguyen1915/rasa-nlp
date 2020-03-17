from haolib.lib_rasa.utils import *
from test_case.nlu.featurizers.utils import *
from rasa.nlu.constants import (
    TEXT,
    RESPONSE,
    SPACY_DOCS,
    TOKENS_NAMES,
)
from rasa.nlu.tokenizers.spacy_tokenizer import SpacyTokenizer
from rasa.nlu.training_data import Message


def test_regex_featurizer():
    """
    Last one is union of value above
    :return:
    """
    from rasa.nlu.featurizers.sparse_featurizer.regex_featurizer import RegexFeaturizer
    sentence, expected, labeled_tokens = (
        "hey how are you today",
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        [0],
    )
    patterns = [
        {"pattern": "[0-9]+", "name": "number", "usage": "intent"},
        {"pattern": "\\bhey*", "name": "hello", "usage": "intent"},
        {"pattern": "[0-1]+", "name": "binary", "usage": "intent"},
    ]
    ftr = RegexFeaturizer({}, known_patterns=patterns)

    # adds tokens to the message
    tokenizer = SpacyTokenizer({})
    message = Message(sentence, data={RESPONSE: sentence})
    assert show_message(message, False) == {
        "response": "hey how are you today",
        "text": "hey how are you today"
    }
    message.set(SPACY_DOCS[TEXT], spacy_nlp(sentence))
    tokenizer.process(message)
    # assert show_message(message) == {'response': 'hey how are you today', 'text_spacy_doc': spacy_nlp("hey how are you today"),
    #                                  'tokens': ['hey', 'how', 'are', 'you', 'today', '__CLS__'],
    #                                  'text': 'hey how are you today'}
    # result = ftr._features_for_patterns(message, TEXT)
    ftr.process(message)  # [TEXT, RESPONSE]
    show_message(message)
    assert len(message.get(TOKENS_NAMES[TEXT], [])) > 0


if __name__ == "__main__":
    test_regex_featurizer()
