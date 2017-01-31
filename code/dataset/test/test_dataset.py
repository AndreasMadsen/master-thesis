
from nose.tools import *

import math

import sugartensor as stf

from dataset.synthetic_digits import SyntheticDigits


def tensorflow_extract(dataset):
    with stf.Session() as sess:
        stf.sg_init(sess)
        with stf.sg_queue_context():
            sources, targets = sess.run([
                dataset.source, dataset.target
            ])

    return list(zip(
        dataset.decode_as_batch(sources),
        dataset.decode_as_batch(targets)
    ))


def test_output_order():
    """ensure that Dataset order is consistent"""
    dataset = SyntheticDigits(
        examples=10, seed=99, shuffle=False
    )
    actual = list(dataset)

    assert_equal(actual, [
        ('three eight eight', '388'),
        ('eight four nine', '849'),
        ('two eight', '28'),
        ('five four', '54'),
        ('seven three zero', '730'),
        ('seven zero four', '704'),
        ('one one nine', '119'),
        ('four one', '41'),
        ('six seven one', '671'),
        ('four zero seven', '407')
    ])


def test_all_examples_exposed():
    """ensure all examples are exposed"""
    dataset = SyntheticDigits(
        examples=10, seed=99, shuffle=False
    )
    actual = tensorflow_extract(dataset)
    expected = list(map(lambda v: (f'{v[0]}#', f'{v[1]}#'), dataset))

    actual = sorted(actual, key=lambda v: v[1])
    expected = sorted(expected, key=lambda v: v[1])

    assert_equal(actual, expected)
