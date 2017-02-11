
from nose.tools import assert_equal

import numpy as np
import tensorflow as tf
import sugartensor as stf

from code.tf_operator.cross_entropy.cross_entropy_indirect \
    import cross_entropy_indirect


def test_output_shape():
    "validate output shape of cross_entropy_indirect()"
    with tf.Session() as sess:
        num_batch = 10

        scaled_logits = tf.zeros((num_batch,), dtype=tf.float32)
        loss = cross_entropy_indirect(scaled_logits, name="cross_entropy")

        stf.sg_init(sess)
        assert_equal(sess.run(loss).shape, tuple())
