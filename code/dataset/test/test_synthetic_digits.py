
from nose.tools import assert_equal

import sugartensor as stf

from code.dataset.synthetic_digits import SyntheticDigits


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
    """ensure that SyntheticDigits order is consistent"""
    dataset = SyntheticDigits(
        batch_size=10, examples=10,
        shuffle=False, seed=99, repeat=False,
        tqdm=False
    )
    assert_equal(list(dataset), [
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


def test_properties():
    """ensure that SyntheticDigits vocabilary and observations is correct"""
    dataset = SyntheticDigits(
        examples=1, seed=99,
        tqdm=False
    )

    assert_equal(
        dataset.properties.vocabulary,
        frozenset('zerontwhufivsxg0123456789 ')
    )
    assert_equal(dataset.properties.histogram.observations, 1)
