
from nose.tools import assert_equal

from code.moses.multi_bleu import multi_bleu


def test_multi_bleu():
    """test moses multi-blue"""

    tokenized = [
        'hello world , how are you ?',
        'I \'m fine , thank you !'
    ]

    assert_equal(
        multi_bleu(tokenized, tokenized),
        'BLEU = 100.00, 100.0/100.0/100.0/100.0' +
        ' (BP=1.000, ratio=1.000, hyp_len=14, ref_len=14)'
    )
