
import tensorflow as tf


def batch_gather(tensor, indices, name=None):
    with tf.name_scope(name, 'batch-gather', values=[tensor, indices]):
        batch_indices = tf.range(tf.shape(indices, out_type=indices.dtype)[0])
        gather_indices = tf.stack([batch_indices, indices], -1)

        collect = tf.gather_nd(tensor, gather_indices)
        collect.set_shape(
            indices.get_shape().concatenate(tensor.get_shape()[2:])
        )

        return collect
