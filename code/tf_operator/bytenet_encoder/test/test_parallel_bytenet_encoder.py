
from nose.tools import assert_equal

import tensorflow as tf
import sugartensor as stf

from code.tf_operator.bytenet_encoder.parallel_bytenet_encoder \
    import parallel_bytenet_encoder


def test_output_shape():
    "validate output shape of parallel_bytenet_encoder()"
    with tf.Session() as sess:
        (num_batch, seq_len, num_dims) = (10, 15, 8)

        enc = tf.zeros((num_batch, seq_len, num_dims))
        enc = parallel_bytenet_encoder(enc)

        stf.sg_init(sess)
        assert_equal(sess.run(enc).shape, (num_batch, seq_len, num_dims))
