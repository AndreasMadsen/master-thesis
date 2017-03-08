
import tensorflow as tf


def batch_beam_gather(tensor, indices, name=None):
    with tf.name_scope(name, 'batch-beam-gather', values=[tensor, indices]):
        beam_size = int(indices.get_shape()[1])

        batch_indices = tf.range(tf.shape(indices, out_type=indices.dtype)[0])
        batch_indices = tf.expand_dims(batch_indices, -1)
        batch_indices = tf.tile(batch_indices, [1, beam_size])

        gather_indices = tf.stack([batch_indices, indices], -1)

        collect = tf.gather_nd(tensor, gather_indices)
        collect.set_shape(
            indices.get_shape().concatenate(tensor.get_shape()[2:])
        )

        return collect
