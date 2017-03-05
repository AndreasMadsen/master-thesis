
from nose.tools import assert_equal

import numpy as np
import tensorflow as tf
import sugartensor as stf

from code.metric.out_of_bound import OutOfBound
from code.dataset import SyntheticDigits
from code.metric.test.dummy_model import DummyModel


def test_out_of_bound():
    """test out of bound metric"""
    dataset = SyntheticDigits(batch_size=16, tqdm=False)

    # encode translated text
    translated = np.asarray([
        [0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 28, 1, 2]
    ])
    translated = np.tile(translated, (32, 1))

    # setup model
    model = DummyModel(dataset, translated)
    oob = OutOfBound(dataset).build(model)

    with tf.Session() as sess:
        stf.sg_init(sess)
        with stf.sg_queue_context():
            assert_equal(sess.run(oob), 16)
