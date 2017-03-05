
from nose.tools import assert_equal

import numpy as np
import sugartensor as stf

from code.dataset.synthetic_digits import SyntheticDigits


def tensorflow_extract_one(dataset):
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


def tensorflow_extract_all(dataset):
    all_sources = []
    all_targets = []

    with stf.Session() as sess:
        stf.sg_init(sess)
        with stf.sg_queue_context():

            for i in range(dataset.num_batch):
                sources, targets = sess.run([
                    dataset.source, dataset.target
                ])

                all_sources += dataset.decode_as_batch(sources)
                all_targets += dataset.decode_as_batch(targets)

    return list(zip(all_sources, all_targets))


def test_all_examples_exposed_one():
    """ensure all examples are exposed when batch_size = observations"""
    dataset = SyntheticDigits(
        batch_size=11, examples=11,
        seed=99, shuffle=False, repeat=False,
        tqdm=False
    )
    actual = tensorflow_extract_one(dataset)
    expected = list(map(lambda v: (f'{v[0]}^', f'{v[1]}^'), dataset))

    actual = sorted(actual, key=lambda v: v[1])
    expected = sorted(expected, key=lambda v: v[1])

    assert_equal(actual, expected)


def test_all_examples_exposed_all():
    """ensure all examples are exposed when batch_size < observations"""
    dataset = SyntheticDigits(
        batch_size=2, examples=11,
        seed=99, shuffle=False, repeat=False,
        tqdm=False
    )
    actual = tensorflow_extract_all(dataset)
    expected = list(map(lambda v: (f'{v[0]}^', f'{v[1]}^'), dataset))

    actual = sorted(actual, key=lambda v: v[1])
    expected = sorted(expected, key=lambda v: v[1])

    assert_equal(actual, expected)


def test_decode_encoding():
    """test special chars in decoding algorithm"""
    dataset = SyntheticDigits(
        batch_size=2, examples=11, seed=99,
        shuffle=False, repeat=False,
        tqdm=False
    )

    assert_equal(
        dataset.decode_as_str(
            np.asarray([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 28, 1, 2])
        ),
        '⨯0123456789�^'
    )

    assert_equal(
        dataset.decode_as_str(
            np.asarray([0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 28, 1, 2]),
            show_eos=False
        ),
        '⨯0123456789�'
    )
