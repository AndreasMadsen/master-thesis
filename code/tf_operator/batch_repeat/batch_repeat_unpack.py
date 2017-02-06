
import tensorflow as tf


def batch_repeat_unpack(x, repeats=1, name=None):
    with tf.name_scope(name, "batch-repeat-unpack", values=[x]):
        # x.shape = (batches, repeats, ...)

        # reshape to (batches * repeats, ...)
        shape = tf.concat_v2([[-1], [repeats], tf.shape(x)[1:]], axis=0)
        t = tf.reshape(x, shape=shape)

        return t
