
from nose.tools import assert_equal

from code.moses.tokenizer import Tokenizer


def test_tokenizer():
    """test moses tokenizer"""

    with Tokenizer('en') as tokenizer:
        tokenizer.write('hello world, how are you?')
        tokenizer.write('I\'m fine, thank you!')

    assert_equal(
        list(tokenizer),
        [
            'hello world , how are you ?',
            'I \'m fine , thank you !'
        ]
    )
