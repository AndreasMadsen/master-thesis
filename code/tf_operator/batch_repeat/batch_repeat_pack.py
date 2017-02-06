
import tensorflow as tf


def batch_repeat_pack(x, name=None):
    with tf.name_scope(name, "batch-repeat-pack", values=[x]):
        # x.shape = (batches, repeats, ...)

        # reshape to (batches * repeats, ...)
        shape = tf.concat_v2([[-1], tf.shape(x)[2:]], axis=0)
        t = tf.reshape(x, shape=shape)

        return t
