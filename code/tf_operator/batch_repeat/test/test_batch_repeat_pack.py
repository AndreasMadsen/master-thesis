
from nose.tools import *

import numpy as np
import tensorflow as tf

from code.tf_operator.batch_repeat.batch_repeat_pack import batch_repeat_pack


def test_output_scalar():
    "validate output of batch_repeat_pack() on scalar values"
    with tf.Session() as sess:
        data = np.asarray([[1, 1], [2, 2], [3, 3]], dtype=np.float32)
        tensor = tf.placeholder(shape=[None] * 2, name='x', dtype=tf.float32)

        np.testing.assert_almost_equal(
            sess.run(batch_repeat_pack(tensor), feed_dict={tensor: data}),
            np.asarray([
                1, 1, 2, 2, 3, 3
            ])
        )


def test_output_vector():
    "validate output of batch_repeat_pack() on vector values"
    with tf.Session() as sess:
        data = np.asarray([
            [[1, 2, 3], [1, 2, 3]],
            [[4, 5, 6], [4, 5, 6]],
            [[7, 8, 9], [7, 8, 9]]
        ], dtype=np.float32)
        tensor = tf.placeholder(shape=[None] * 3, name='x', dtype=tf.float32)

        np.testing.assert_almost_equal(
            sess.run(batch_repeat_pack(tensor), feed_dict={tensor: data}),
            np.asarray([
                [1, 2, 3], [1, 2, 3],
                [4, 5, 6], [4, 5, 6],
                [7, 8, 9], [7, 8, 9]
            ])
        )
