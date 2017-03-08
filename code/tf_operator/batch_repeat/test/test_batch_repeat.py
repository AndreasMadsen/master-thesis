
from nose.tools import assert_equal

import numpy as np
import tensorflow as tf

from code.tf_operator.batch_repeat.batch_repeat import batch_repeat


def test_output_vector():
    "validate output of batch_repeat() on vector"
    with tf.Session() as sess:
        data = np.asarray([1, 2, 3], dtype=np.float32)
        tensor = tf.placeholder(shape=[None], name='x', dtype=tf.float32)
        repeat = batch_repeat(tensor, 2)

        assert_equal(repeat.get_shape().as_list(), [None, 2])
        np.testing.assert_almost_equal(
            sess.run(repeat, feed_dict={tensor: data}),
            np.asarray([
                [data[0], data[0]],
                [data[1], data[1]],
                [data[2], data[2]]
            ])
        )


def test_output_matrix():
    "validate output of batch_repeat() on matrix"
    with tf.Session() as sess:
        data = np.asarray([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.float32)
        tensor = tf.placeholder(shape=[None] * 2, name='x', dtype=tf.float32)
        repeat = batch_repeat(tensor, 2)

        assert_equal(repeat.get_shape().as_list(), [None, 2, None])
        np.testing.assert_almost_equal(
            sess.run(repeat, feed_dict={tensor: data}),
            np.asarray([
                [data[0], data[0]],
                [data[1], data[1]],
                [data[2], data[2]]
            ])
        )


def test_output_tensor():
    "validate output of batch_repeat() on tensor"
    with tf.Session() as sess:
        data = np.asarray([
            [[11, 12, 13],
             [14, 15, 16],
             [17, 18, 19]],
            [[21, 22, 23],
             [24, 25, 26],
             [27, 28, 29]],
            [[31, 32, 33],
             [34, 35, 36],
             [37, 38, 39]]
        ], dtype=np.float32)
        tensor = tf.placeholder(
            shape=[None, None, 3], name='x', dtype=tf.float32
        )
        repeat = batch_repeat(tensor, 2)

        assert_equal(repeat.get_shape().as_list(), [None, 2, None, 3])
        np.testing.assert_almost_equal(
            sess.run(repeat, feed_dict={tensor: data}),
            np.asarray([
                [data[0], data[0]],
                [data[1], data[1]],
                [data[2], data[2]]
            ])
        )
