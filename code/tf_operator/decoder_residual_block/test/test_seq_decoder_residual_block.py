
from nose.tools import *

import sugartensor as stf
import tensorflow as tf
import numpy as np

from code.tf_operator.decoder_residual_block.seq_decoder_residual_block \
    import seq_decoder_residual_block, seq_decoder_residual_block_init
from code.tf_operator.decoder_residual_block.parallel_decoder_residual_block \
    import parallel_decoder_residual_block


def test_equal_output():
    """validate seq_decoder_residual_block() matches res_block(causal=True)"""

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
        with tf.variable_scope('seq-causal-res-block-test') as weight_scope:
            with tf.variable_scope('activation') as _:
                stf.sg_initializer.constant(
                    'beta', dims, summary=False
                )
                stf.sg_initializer.constant(
                    'gamma', dims, value=1, summary=False
                )

            with tf.variable_scope('reduce-dim') as _:
                stf.sg_initializer.constant(
                    'beta', dims // 2, summary=False
                )
                stf.sg_initializer.constant(
                    'gamma', dims // 2, value=1, summary=False
                )
                stf.sg_initializer.he_uniform(
                    'W', (1, dims, dims // 2)
                )

            with tf.variable_scope('conv-dilated') as _:
                stf.sg_initializer.constant(
                    'beta', dims // 2, summary=False
                )
                stf.sg_initializer.constant(
                    'gamma', dims // 2, value=1, summary=False
                )
                stf.sg_initializer.he_uniform(
                    'W', (1, size, dims // 2, dims // 2)
                )

            with tf.variable_scope('recover-dim') as _:
                stf.sg_initializer.he_uniform(
                    'W', (1, dims // 2, dims)
                )
                stf.sg_initializer.constant(
                    'b', dims
                )

        # reference implementation
        conv1d_output = parallel_decoder_residual_block(
            tensor,
            size=size, rate=rate,
            name=weight_scope, reuse=True
        )

        # apply seq_decoder_residual_block to all time steps
        def scan_op(acc, xt):
            (previous, _) = acc
            previous, out = seq_decoder_residual_block(
                xt, previous, size=size, rate=rate,
                name=weight_scope, reuse=True
            )
            return (previous, out)

        (_, seq_output) = tf.scan(
            scan_op,
            elems=tf.transpose(tensor, perm=[1, 0, 2]),
            initializer=(
                seq_decoder_residual_block_init(tensor, size=size, rate=rate),
                tf.zeros((tf.shape(tensor)[0], dims))
            )
        )
        seq_output = tf.transpose(seq_output, perm=[1, 0, 2])

        # initialize weights
        stf.sg_init(sess)

        # check equality
        np.testing.assert_almost_equal(
            sess.run(conv1d_output, feed_dict={tensor: data}),
            sess.run(seq_output, feed_dict={tensor: data})
        )
