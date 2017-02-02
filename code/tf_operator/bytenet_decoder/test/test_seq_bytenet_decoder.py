
from nose.tools import *

import numpy as np
import tensorflow as tf
import sugartensor as stf

from code.tf_operator.bytenet_decoder.parallel_bytenet_decoder \
    import parallel_bytenet_decoder
from code.tf_operator.bytenet_decoder.seq_bytenet_decoder \
    import seq_bytenet_decoder_init, seq_bytenet_decoder


def test_output_shape():
    """validate seq_bytenet_decoder() matches parallel_bytenet_decoder()"""
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

        # reference implementation
        parallel_output = parallel_bytenet_decoder(
            tensor.sg_concat(target=tensor), name="decoder"
        )

        # initalize scan state
        init_state = seq_bytenet_decoder_init(tensor)

        # apply seq_decoder_residual_block to all time steps
        def scan_op(acc, enc_t):
            (state_tm1, y_tm1) = acc

            # concat with itself, like in the parallel case
            val = enc_t.sg_concat(target=enc_t)

            # decode graph ( causal convolution )
            state_t, out = seq_bytenet_decoder(
                state_tm1, val, name="decoder", reuse=True
            )

            return (state_t, out)

        (_, seq_output) = tf.scan(
            scan_op,
            elems=tf.transpose(tensor, perm=[1, 0, 2]),
            initializer=(
                init_state,
                tf.zeros(
                    (tf.shape(tensor)[0], 2 * dims), dtype=tensor.dtype
                )  # labels
            )
        )

        seq_output = tf.transpose(seq_output, perm=[1, 0, 2])

        # initialize weights
        stf.sg_init(sess)

        # check equality
        np.testing.assert_almost_equal(
            sess.run(parallel_output, feed_dict={tensor: data}),
            sess.run(seq_output, feed_dict={tensor: data})
        )
