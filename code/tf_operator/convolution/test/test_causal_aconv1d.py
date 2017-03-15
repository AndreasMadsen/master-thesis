
import sugartensor as stf
import tensorflow as tf
import numpy as np

from code.tf_operator.convolution.causal_aconv1d import causal_aconv1d


def test_equal_output():
    """validate causal_aconv1d(lowmem) matches causal_aconv1d(highmem)"""

    batch_size = 16
    time_steps = 8
    size = 3
    rate = 2
    dims = 10

    data = np.random.uniform(-3, 3, size=(batch_size, time_steps, dims))

    with tf.Session() as sess:
        tensor = tf.placeholder(
            tf.float32, shape=(None, None, dims), name='x'
        )

        # intialize parameters
        with tf.variable_scope('seq-causal-aconv1d-low-test') as weight_scope:
            stf.sg_initializer.he_uniform('W', (size, dims, dims))
            stf.sg_initializer.constant('b', dims)

        # reference implementation
        high_output = causal_aconv1d(
            tensor,
            size=size, rate=rate,
            name=weight_scope, reuse=True
        )

        low_output = causal_aconv1d(
            tensor,
            size=size, rate=rate,
            low_memory=True,
            name=weight_scope, reuse=True
        )

        # initialize weights
        stf.sg_init(sess)

        # check equality
        np.testing.assert_almost_equal(
            sess.run(high_output, feed_dict={tensor: data}),
            sess.run(low_output, feed_dict={tensor: data})
        )
