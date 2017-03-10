
import tensorflow as tf


def batch_repeat(x, repeats=1, name=None):
    with tf.name_scope(name, "batch-repeat", values=[x]):
        x_dims = len(x.get_shape())

        # reshape into (batches, 1, ....)
        t = tf.expand_dims(x, axis=1)

        # transpose into (batches, repeats, ....)
        multiples = [1, repeats] + [1] * (x_dims - 1)
        t = tf.tile(t, multiples=multiples)

        return t
