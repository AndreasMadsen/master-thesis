
from nose.tools import assert_equal

import numpy as np
import tensorflow as tf

from code.tf_operator.select_value.batch_gather import batch_gather


def test_output_tensor():
    "validate output of batch_gather() on tensor"
    with tf.Session() as sess:
        data = np.asarray([
            [[11, 12, 13],
             [14, 15, 16],
             [17, 18, 19]],
            [[21, 22, 23],
             [24, 25, 26],
             [27, 28, 29]],
            [[31, 32, 33],
             [34, 35, 36],
             [37, 38, 39]]
        ], dtype=np.float32)
        indices = np.asarray([
            0,
            1,
            1
        ])
        d_tensor = tf.placeholder(shape=[None, 3, 3], name='x',
                                  dtype=tf.float32)
        i_tensor = tf.placeholder(shape=[None], name='i', dtype=tf.int64)
        out = batch_gather(d_tensor, i_tensor)

        assert_equal(out.get_shape().as_list(), [None, 3])
        np.testing.assert_almost_equal(
            sess.run(
                out,
                feed_dict={d_tensor: data, i_tensor: indices}
            ),
            np.asarray([
                [11, 12, 13],
                [24, 25, 26],
                [34, 35, 36]
            ], dtype=np.float32)
        )
