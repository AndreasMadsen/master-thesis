
from nose.tools import assert_equal

import numpy as np
import tensorflow as tf
import sugartensor as stf

from code.tf_operator.decoder_residual_block.parallel_decoder_residual_block \
    import parallel_decoder_residual_block


def test_output_shape():
    "validate output shape of parallel_decoder_residual_block()"
    with tf.Session() as sess:
        (num_batch, seq_len, num_dims) = (10, 15, 8)

        enc = tf.zeros((num_batch, seq_len, num_dims))
        enc = parallel_decoder_residual_block(enc)

        stf.sg_init(sess)
        assert_equal(enc.get_shape(), (num_batch, seq_len, num_dims))
        assert_equal(sess.run(enc).shape, (num_batch, seq_len, num_dims))


def test_output_shape_low_memory():
    "validate output shape of parallel_decoder_residual_block(lowmem)"
    with tf.Session() as sess:
        (num_batch, seq_len, num_dims) = (10, 15, 8)

        enc = tf.zeros((num_batch, seq_len, num_dims))
        enc = parallel_decoder_residual_block(enc, low_memory=True)

        stf.sg_init(sess)
        assert_equal(enc.get_shape(), (num_batch, seq_len, num_dims))
        assert_equal(sess.run(enc).shape, (num_batch, seq_len, num_dims))


def test_output_low_memory():
    "validate output parallel_decoder_residual_block(lowmem) match highmem"
    with tf.Session() as sess:
        (num_batch, seq_len, num_dims) = (10, 15, 8)

        enc = tf.zeros((num_batch, seq_len, num_dims))
        enc_hi_mem = parallel_decoder_residual_block(enc,
                                                     name="resblock-memory")
        enc_lo_mem = parallel_decoder_residual_block(enc, low_memory=True,
                                                     name="resblock-memory",
                                                     reuse=True)

        stf.sg_init(sess)
        np.testing.assert_almost_equal(
            sess.run(enc_hi_mem),
            sess.run(enc_lo_mem)
        )
