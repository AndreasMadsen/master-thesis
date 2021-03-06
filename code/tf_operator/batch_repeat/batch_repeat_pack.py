
import tensorflow as tf


def batch_repeat_pack(x, name=None):
    with tf.name_scope(name, "batch-repeat-pack", values=[x]):
        # x.shape = (batches, repeats, ...)

        # reshape to (batches * repeats, ...)
        shape = tf.concat([[-1], tf.shape(x)[2:]], axis=0)
        t = tf.reshape(x, shape=shape)
        # restore .get_shape() dimentions of output
        t.set_shape(
            tf.TensorShape([
                x.get_shape()[0] * x.get_shape()[1]
            ]).concatenate(x.get_shape()[2:])
        )

        return t
