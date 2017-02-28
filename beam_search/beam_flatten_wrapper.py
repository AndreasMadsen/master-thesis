
import tensorflow as tf
from tensorflow.python.util import nest


def nest_map(func, nested):
    flat = nest.flatten(nested)
    return nest.pack_sequence_as(nested, list(map(func, flat)))


class BeamFlattenWrapper(tf.nn.rnn_cell.RNNCell):

    def __init__(self, cell, beam_size):
        self.cell = cell
        self.beam_size = beam_size

    def merge_batch_beam(self, tensor):
        new_shape = tf.concat(0, [[-1], tf.shape(tensor)[2:]])
        res = tf.reshape(tensor, new_shape)
        res.set_shape(
            tf.TensorShape((None,))
              .concatenate(tensor.get_shape()[2:])
        )
        return res

    def unmerge_batch_beam(self, tensor):
        new_shape = tf.concat(0, [[-1, self.beam_size], tf.shape(tensor)[1:]])
        res = tf.reshape(tensor, new_shape)
        res.set_shape(
            tf.TensorShape((None, self.beam_size))
              .concatenate(tensor.get_shape()[1:])
        )
        return res

    def prepend_beam_size(self, element):
        return tf.TensorShape(self.beam_size).concatenate(element)

    def tile_along_beam(self, state):
        if nest.is_sequence(state):
            return nest_map(self.tile_along_beam, state)

        if not isinstance(state, tf.Tensor):
            raise ValueError("State should be a sequence or tensor")

        tensor = state

        # [batch, beam_size, rest ...]
        tensor_shape = tensor.get_shape().with_rank_at_least(1)
        new_tensor_shape = tensor_shape[:1].concatenate(self.beam_size) \
                                           .concatenate(tensor_shape[1:])

        res = tf.expand_dims(tensor, 1)
        res = tf.tile(res, [1, self.beam_size] + [1] * (tensor_shape.ndims - 1))
        res.set_shape(new_tensor_shape)
        return res

    def __call__(self, inputs, state, scope=None):
        flat_inputs = nest_map(self.merge_batch_beam, inputs)
        flat_state = nest_map(self.merge_batch_beam, state)

        flat_output, flat_next_state = self.cell(
            flat_inputs, flat_state, scope=scope
        )

        output = nest_map(self.unmerge_batch_beam, flat_output)
        next_state = nest_map(self.unmerge_batch_beam, flat_next_state)

        return output, next_state

    @property
    def state_size(self):
        return nest_map(self.prepend_beam_size, self.cell.state_size)

    @property
    def output_size(self):
        return nest_map(self.prepend_beam_size, self.cell.output_size)
