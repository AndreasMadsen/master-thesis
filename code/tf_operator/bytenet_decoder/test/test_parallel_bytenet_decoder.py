
from nose.tools import *

import tensorflow as tf
import sugartensor as stf

from code.tf_operator.bytenet_decoder.parallel_bytenet_decoder \
    import parallel_bytenet_decoder


def test_output_shape():
    "validate output shape of parallel_bytenet_decoder()"
    with tf.Session() as sess:
        (num_batch, seq_len, num_dims) = (10, 15, 8)

        enc = tf.zeros((num_batch, seq_len, num_dims))
        enc = parallel_bytenet_decoder(enc)

        stf.sg_init(sess)
        assert_equal(sess.run(enc).shape, (num_batch, seq_len, num_dims))
