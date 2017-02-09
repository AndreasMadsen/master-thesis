
import sugartensor as stf
import tensorflow as tf
import numpy as np

from code.tf_operator.decoder_residual_block.seq_causal_aconv1d \
    import seq_causal_aconv1d


def test_equal_output():
    """validate seq_causal_aconv1d() matches aconv1d(causal=True)"""

    batch_size = 16
    time_steps = 8
    size = 3
    rate = 2
    in_dims = 10
    out_dims = in_dims // 2

    data = np.random.uniform(-3, 3, size=(batch_size, time_steps, in_dims))

    with tf.Session() as sess:
        tensor = tf.placeholder(
            tf.float32, shape=(None, None, in_dims), name='x'
        )

        # intialize parameters
        with tf.variable_scope('seq-causal-aconv1d-test') as weight_scope:
            stf.sg_initializer.he_uniform('W', (1, size, in_dims, out_dims))
            stf.sg_initializer.constant('b', out_dims)

        # reference implementation
        conv1d_output = tensor.sg_aconv1d(
            dim=out_dims, size=size, rate=rate,
            causal=True,
            name=weight_scope, reuse=True
        )

        # apply seq_dim_reduction to all time steps
        zero_initializer = tf.zeros((tf.shape(tensor)[0], in_dims))
        zero_padding = [zero_initializer for i in range((size - 1) * rate)]

        def scan_op(acc, xt):
            (previous, _) = acc
            out = seq_causal_aconv1d(
                xt, previous=previous, dim=out_dims, size=size, rate=rate,
                name=weight_scope, reuse=True
            )
            return ([xt] + previous[:-1], out)

        (_, seq_output) = tf.scan(
            scan_op,
            elems=tf.transpose(tensor, perm=[1, 0, 2]),
            initializer=(
                zero_padding,
                tf.zeros((tf.shape(tensor)[0], out_dims))
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
