
import tensorflow as tf


def batch_repeat(x, repeats=1, name=None, reuse=False):
    with tf.name_scope(name, "bytenet-decoder", values=[x]):
        x_dims = len(x.get_shape())

        # repeat on the first dimention, (batches * repeats, ...)
        multiples = [1] * x_dims
        multiples[0] = repeats
        t = tf.tile(x, multiples=multiples)

        # reshape into (repeats, batches, ....)
        shape = tf.concat_v2([[repeats], tf.shape(x)], axis=0)
        t = tf.reshape(t, shape=shape)

        # transpose into (batches, repeats, ....)
        perm = list(range(x_dims + 1))
        perm[0] = 1
        perm[1] = 0
        t = tf.transpose(t, perm=perm)

        return t
