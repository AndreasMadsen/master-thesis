
from nose.tools import assert_equal

import os.path as path

from code.moses.tokenizer import Tokenizer
from code.moses.multi_bleu import multi_bleu


def _load_fixture(name):
    # read google translated lines
    this_dir = path.dirname(path.realpath(__file__))
    filepath = path.join(this_dir, f'fixtures/{name}.txt')
    with open(filepath, encoding='utf-8') as file:
        translated_lines = [line.rstrip() for line in file]
    return translated_lines


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


def test_multi_bleu_from_tokenizer():
    target_list = _load_fixture('target.en')
    translation_list = _load_fixture('translation.en')

    target_tokenizer = Tokenizer('en')
    translation_tokenizer = Tokenizer('en')

    with target_tokenizer, translation_tokenizer:
        for target, translation in zip(target_list, translation_list):
            target_tokenizer.write(target)
            translation_tokenizer.write(translation)

    assert_equal(
        multi_bleu(translate=translation_tokenizer, target=target_tokenizer),
        'BLEU = 0.97, 29.5/3.5/0.8/0.2' +
        ' (BP=0.469, ratio=0.569, hyp_len=26643, ref_len=46833)'
    )
