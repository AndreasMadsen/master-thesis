
from nose.tools import assert_equal

import tensorflow as tf
import sugartensor as stf

from code.tf_operator.cross_entropy.cross_entropy_direct \
    import cross_entropy_direct


def test_output_shape():
    "validate output shape of cross_entropy_direct()"
    with tf.Session() as sess:
        (num_batch, seq_len, num_dims) = (10, 15, 8)

        logits = tf.zeros((num_batch, seq_len, num_dims), dtype=tf.float32)
        target = tf.zeros((num_batch, seq_len), dtype=tf.int32)
        loss = cross_entropy_direct(logits, target, name="cross_entropy")

        stf.sg_init(sess)
        assert_equal(sess.run(loss).shape, tuple())
