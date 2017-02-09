
import sugartensor as stf
import tensorflow as tf
import numpy as np

from code.tf_operator.seq_dense.seq_dense import seq_dense


def test_equal_output():
    """validate seq_dense() matches conv1d(size=1)"""

    batch_size = 16
    time_steps = 8
    in_dims = 10
    out_dims = in_dims // 2

    data = np.random.uniform(-3, 3, size=(batch_size, time_steps, in_dims))

    with tf.Session() as sess:
        tensor = tf.placeholder(
            tf.float32, shape=(None, None, in_dims), name='x'
        )

        # intialize parameters
        with tf.variable_scope('seq-dense-test') as weight_scope:
            stf.sg_initializer.he_uniform('W', (1, in_dims, out_dims))
            stf.sg_initializer.constant('b', out_dims)

        # reference implementation
        conv1d_output = tensor.sg_conv1d(
            size=1, dim=out_dims,
            name=weight_scope, reuse=True
        )

        # apply seq_dim_reduction to all time steps
        seq_output = tf.scan(
            lambda _, xt: seq_dense(
                xt, dim=out_dims,
                name=weight_scope, reuse=True
            ),
            elems=tf.transpose(tensor, perm=[1, 0, 2]),
            initializer=tf.zeros(
                (tf.shape(tensor)[0], out_dims), dtype=tensor.dtype
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
