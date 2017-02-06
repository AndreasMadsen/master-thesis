
from nose.tools import *

import numpy as np
import tensorflow as tf

from code.tf_operator.seq_prop import seq_prop


def test_output_unmasked():
    "validate output of seq_prop(mask=None)"
    with tf.Session() as sess:
        data = np.asarray([
            [0.1, 0.1, 0.1],
            [0.5, 0.5, 0.2],
            [0.1, 0.1, 0.2]
        ], dtype=np.float32)
        tensor = tf.placeholder(shape=[None] * 2, name='x', dtype=tf.float32)

        np.testing.assert_almost_equal(
            sess.run(seq_prop(tensor, axis=1), feed_dict={tensor: data}),
            np.asarray([
                0.001, 0.05, 0.002
            ])
        )


def test_output_masked():
    "validate output of seq_prop(mask=tf.Tensor)"
    with tf.Session() as sess:
        data = np.asarray([
            [0.1, 0.1, 0.1],
            [0.5, 0.5, 0.2],
            [0.1, 0.1, 0.2]
        ], dtype=np.float32)
        mask = np.asarray([
            [3, 2, 1],
            [1, 0, 0],
            [1, 1, 0]
        ], dtype=np.float32)

        d_tensor = tf.placeholder(shape=[None] * 2, name='x', dtype=tf.float32)
        m_tensor = tf.placeholder(shape=[None] * 2, name='m', dtype=tf.int32)

        np.testing.assert_almost_equal(
            sess.run(
                seq_prop(d_tensor, mask=m_tensor, axis=1),
                feed_dict={d_tensor: data, m_tensor: mask}
            ),
            np.asarray([
                0.001, 0.5, 0.01
            ])
        )
