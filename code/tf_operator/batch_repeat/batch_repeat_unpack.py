
import tensorflow as tf


def batch_repeat_unpack(x, repeats=1, name=None):
    with tf.name_scope(name, "batch-repeat-unpack", values=[x]):
        # x.shape = (batches, repeats, ...)

        # reshape to (batches * repeats, ...)
        shape = tf.concat([[-1], [repeats], tf.shape(x)[1:]], axis=0)
        t = tf.reshape(x, shape=shape)

        repeats_dim = tf.Dimension(repeats)
        t.set_shape(
            tf.TensorShape([
                x.get_shape()[0] // repeats_dim, repeats_dim
            ]).concatenate(x.get_shape()[1:])
        )

        return t
