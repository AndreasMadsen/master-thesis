
from nose.tools import *

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
        assert_equal(sess.run(enc).shape, (num_batch, seq_len, num_dims))
