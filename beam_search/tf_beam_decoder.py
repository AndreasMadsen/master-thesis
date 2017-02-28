"""
Beam decoder for tensorflow

Sample usage:

```
from tf_beam_decoder import  beam_decoder

decoded_sparse, decoded_logprobs = beam_decoder(
    cell=cell,
    beam_size=7,
    stop_token=2,
    initial_state=initial_state,
    initial_input=initial_input,
)
```

See the `beam_decoder` function for complete documentation. (Only the
`beam_decoder` function is part of the public API here.)
"""
import tensorflow as tf
import numpy as np

from tensorflow.python.util import nest

from beam_flatten_wrapper import BeamFlattenWrapper


def nest_map(func, nested):
    if not nest.is_sequence(nested):
        return func(nested)
    flat = nest.flatten(nested)
    return nest.pack_sequence_as(nested, list(map(func, flat)))


def flat_batch_gather(flat_params, indices, validate_indices=True,
                      batch_size=None,
                      options_size=None):
    """
    Gather slices from `flat_params` according to `indices`, separately for each
    example in a batch.

    output[(b * indices_size + i), :, ..., :] = flat_params[(b * options_size + indices[b, i]), :, ..., :]

    The arguments `batch_size` and `options_size`, if provided, are used instead
    of looking up the shape from the inputs. This may help avoid redundant
    computation (TODO: figure out if tensorflow's optimizer can do this automatically)

    Args:
      flat_params: A `Tensor`, [batch_size * options_size, ...]
      indices: A `Tensor`, [batch_size, indices_size]
      validate_indices: An optional `bool`. Defaults to `True`
      batch_size: an integer or scalar tensor representing the batch size
      options_size: an integer or scalar Tensor representing the number of options to choose from
    """
    indices_offsets = tf.reshape(tf.range(
        batch_size) * options_size, [-1] + [1] * (len(indices.get_shape()) - 1))
    indices_into_flat = indices + tf.cast(indices_offsets, indices.dtype)
    flat_indices_into_flat = tf.reshape(indices_into_flat, [-1])

    return tf.gather(flat_params, flat_indices_into_flat, validate_indices=validate_indices)


def batch_gather(params, indices, validate_indices=True,
                 batch_size=None,
                 options_size=None):
    """
    Gather slices from `params` according to `indices`, separately for each
    example in a batch.

    output[b, i, ..., j, :, ..., :] = params[b, indices[b, i, ..., j], :, ..., :]

    The arguments `batch_size` and `options_size`, if provided, are used instead
    of looking up the shape from the inputs. This may help avoid redundant
    computation (TODO: figure out if tensorflow's optimizer can do this automatically)

    Args:
      params: A `Tensor`, [batch_size, options_size, ...]
      indices: A `Tensor`, [batch_size, ...]
      validate_indices: An optional `bool`. Defaults to `True`
      batch_size: an integer or scalar tensor representing the batch size
      options_size: an integer or scalar Tensor representing the number of options to choose from
    """
    batch_size_times_options_size = batch_size * options_size

    # TODO(nikita): consider using gather_nd. However as of 1/9/2017 gather_nd
    # has no gradients implemented.
    flat_params = tf.reshape(params, tf.concat(
        0, [[batch_size_times_options_size], tf.shape(params)[2:]]))

    indices_offsets = tf.reshape(tf.range(
        batch_size) * options_size, [-1] + [1] * (len(indices.get_shape()) - 1))
    indices_into_flat = indices + tf.cast(indices_offsets, indices.dtype)

    return tf.gather(flat_params, indices_into_flat, validate_indices=validate_indices)


class BeamSearchHelper(object):
    # Our beam scores are stored in a fixed-size tensor, but sometimes the
    # tensor size is greater than the number of elements actually on the beam.
    # The invalid elements are assigned a highly negative score.
    # However, top_k errors if any of the inputs have a score of -inf, so we use
    # a large negative constant instead
    INVALID_SCORE = -1e18

    def __init__(self, cell, beam_size, stop_token, initial_state, initial_input,
                 scope=None, max_len=100
                 ):
        self.beam_size = beam_size
        self.stop_token = stop_token
        self.max_len = max_len
        self.scope = scope

        self.cell = BeamFlattenWrapper(cell, self.beam_size)
        self.initial_state = self.cell.tile_along_beam(initial_state)
        self.initial_input = self.cell.tile_along_beam(initial_input)

        # get the batch size from just one of the tensors in
        # initial_state or initial_input
        batch_size = tf.Dimension(None)
        for tensor in nest.flatten(self.initial_state):
            batch_size = batch_size.merge_with(tensor.get_shape()[0])
        for tensor in nest.flatten(self.initial_input):
            batch_size = batch_size.merge_with(tensor.get_shape()[0])
        self.inferred_batch_size = batch_size.value

        # If the batch size is pre-known then use that as the ShapeSizeOp
        # otherwise default to the tf.shape op.
        if self.inferred_batch_size is not None:
            self.batch_size = self.inferred_batch_size
        else:
            self.batch_size = tf.shape(
                list(nest.flatten(self.initial_state))[0]
            )[0]

        # Precompute the pre-known `batch_size x beam_size`
        self.inferred_batch_size_times_beam_size = None
        if self.inferred_batch_size is not None:
            self.inferred_batch_size_times_beam_size = self.inferred_batch_size * self.beam_size

        # Precompute the `batch_size x beam_size` ShapeOp
        self.batch_size_times_beam_size = self.batch_size * self.beam_size

    def outputs_to_score_fn(self, cell_output):
        return tf.nn.log_softmax(cell_output)

    def beam_setup(self, time):
        emit_output = None
        next_cell_state = self.initial_state
        next_input = self.initial_input

        # Set up the beam search tracking state
        cand_symbols = tf.fill((self.batch_size, 0), self.stop_token)
        cand_logprobs = tf.fill((self.batch_size,), -float('inf'))

        # [0, 1, 2, ..., beam_size, 0, 1, 2, ..., beam_size]
        # [T, F, F, ..., F        , T, F, F, ..., F]
        first_in_beam_mask = tf.equal(
            tf.range(self.batch_size_times_beam_size) % self.beam_size,
            0
        )

        # shape = (batch x beam, 0)
        beam_symbols = tf.fill([self.batch_size_times_beam_size, 0], self.stop_token)
        # [0, I, I, ..., I, 0, I, I, I, ... I]
        beam_logprobs = tf.select(
            first_in_beam_mask,
            tf.fill([self.batch_size_times_beam_size], 0.0),
            tf.fill([self.batch_size_times_beam_size], self.INVALID_SCORE)
        )

        # Set up correct dimensions for maintaining loop invariants.
        # Note that the last dimension (initialized to zero) is not a loop invariant,
        # so we need to clear it. TODO(nikita): is there a public API for clearing shape
        # inference so that _shape is not necessary?
        cand_symbols._shape = tf.TensorShape((self.inferred_batch_size, None))
        cand_logprobs._shape = tf.TensorShape((self.inferred_batch_size,))
        beam_symbols._shape = tf.TensorShape((self.inferred_batch_size_times_beam_size, None))
        beam_logprobs._shape = tf.TensorShape((self.inferred_batch_size_times_beam_size,))

        next_loop_state = (
            cand_symbols,
            cand_logprobs,
            beam_symbols,
            beam_logprobs,
        )

        # shape = (beam, dims)
        emit_output = tf.zeros(self.cell.output_size)
        # shape = (batch, )
        elements_finished = tf.zeros([self.batch_size], dtype=tf.bool)

        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    def beam_loop(self, time, cell_output, cell_state, loop_state):
        (
            past_cand_symbols,  # [batch_size, time-1]
            past_cand_logprobs,  # [batch_size]
            past_beam_symbols,  # [batch_size*beam_size, time-1], right-aligned
            past_beam_logprobs,  # [batch_size*beam_size]
        ) = loop_state

        # PRE
        # reshape to [batch, beam]
        past_beam_logprobs = tf.reshape(past_beam_logprobs, [self.batch_size, self.beam_size])

        # We don't actually use this, but emit_output is required to match the
        # cell output size specfication. Otherwise we would leave this as None.
        emit_output = cell_output

        # 1. Get scores for all candidate sequences
        logprobs = self.outputs_to_score_fn(cell_output)  # [batch, beam, dims]
        num_classes = int(logprobs.get_shape()[-1])

        logprobs_batched = tf.reshape(
            # log(P(dim_i | t) * P(dim_{best} | t - 1)) =
            # [logprobs[batch, beam, dim] + past_beam_logprobs[batch, beam]]
            logprobs + tf.expand_dims(past_beam_logprobs, 2),
            [self.batch_size, self.beam_size * num_classes]
        )

        # 2. Determine which states to pass to next iteration

        # TODO(nikita): consider using slice+fill+concat instead of adding a
        # mask
        # item at index `stop_token` becomes I, rest is 0
        # [0, 0, I, 0, ..., 0]
        nondone_mask = tf.reshape(
            tf.cast(tf.equal(tf.range(num_classes), self.stop_token), tf.float32) * self.INVALID_SCORE,
            [1, 1, num_classes]
        )
        # repeat pattern for each beam:
        # [0, 0, I, 0, ..., 0, 0, 0, I, 0, ..., 0]
        nondone_mask = tf.reshape(
            tf.tile(nondone_mask, [1, self.beam_size, 1]),
            [-1, self.beam_size * num_classes]
        )
        # [x1, x2, I, x3, ..., xD, x1, x2, I, x3, ..., xD]
        logprobs_batched_masked = logprobs_batched + nondone_mask

        # for each batch it returns the beam-most likely labels
        # NOTE: for some reason stop_token is not allowed to be returned
        beam_logprobs, indices = tf.nn.top_k(logprobs_batched_masked, self.beam_size)
        beam_logprobs = tf.reshape(beam_logprobs, [-1])

        # For continuing to the next symbols
        # parent_refs is the beam index for each batch, that resulted in the symbol
        symbols = indices % num_classes  # [batch_size, self.beam_size]
        parent_refs = indices // num_classes  # [batch_size, self.beam_size]

        # replicates past_beam_symbols according to the beam index in parent_refs
        symbols_history = flat_batch_gather(
            past_beam_symbols, parent_refs,
            batch_size=self.batch_size, options_size=self.beam_size
        )
        # concatenate old symbols with new, along the time-dimention
        beam_symbols = tf.concat(1, [symbols_history, tf.reshape(symbols, [-1, 1])])

        # Handle the output and the cell state shuffling
        # also replicate the cell_state according to beam index in parent_refs
        next_cell_state = nest_map(
            lambda element: batch_gather(
                element, parent_refs,
                batch_size=self.batch_size, options_size=self.beam_size
            ),
            cell_state
        )

        next_input = tf.reshape(symbols, [-1, self.beam_size, 1])

        # 3. Update the candidate pool to include entries that just ended with
        # a stop token
        logprobs_done = tf.reshape(logprobs_batched, [-1, self.beam_size, num_classes])[:, :, self.stop_token]
        done_parent_refs = tf.argmax(logprobs_done, 1)
        done_symbols = flat_batch_gather(
            past_beam_symbols, done_parent_refs,
            batch_size=self.batch_size, options_size=self.beam_size
        )

        logprobs_done_max = tf.reduce_max(logprobs_done, 1)

        cand_symbols_unpadded = tf.select(logprobs_done_max > past_cand_logprobs,
                                          done_symbols,
                                          past_cand_symbols)
        cand_logprobs = tf.maximum(logprobs_done_max, past_cand_logprobs)

        cand_symbols = tf.concat(
            1, [cand_symbols_unpadded, tf.fill([self.batch_size, 1], self.stop_token)])

        # 4. Check the stopping criteria
        elements_finished_clip = (time >= self.max_len)
        # This short circuits the search, by using the fact that:
        #  longer sequences can not be more likely than the candidate
        elements_finished_bound = tf.reduce_max(
            tf.reshape(beam_logprobs, [-1, self.beam_size]),
            1
        ) < cand_logprobs
        elements_finished = elements_finished_clip | elements_finished_bound

        # 5. Prepare return values
        # While loops require strict shape invariants, so we manually set shapes
        # in case the automatic shape inference can't calculate these. Even when
        # this is redundant is has the benefit of helping catch shape bugs.

        for tensor in list(nest.flatten(next_input)) + list(nest.flatten(next_cell_state)):
            tensor.set_shape(tf.TensorShape(
                (self.inferred_batch_size, self.beam_size)).concatenate(tensor.get_shape()[2:]))

        for tensor in [cand_symbols, cand_logprobs, elements_finished]:
            tensor.set_shape(tf.TensorShape(
                (self.inferred_batch_size,)).concatenate(tensor.get_shape()[1:]))

        for tensor in [beam_symbols, beam_logprobs]:
            tensor.set_shape(tf.TensorShape(
                (self.inferred_batch_size_times_beam_size,)).concatenate(tensor.get_shape()[1:]))

        next_loop_state = (
            cand_symbols,
            cand_logprobs,
            beam_symbols,
            beam_logprobs,
        )

        return (elements_finished, next_input, next_cell_state,
                emit_output, next_loop_state)

    def loop_fn(self, time, cell_output, cell_state, loop_state):
        if cell_output is None:
            return self.beam_setup(time)
        else:
            return self.beam_loop(time, cell_output, cell_state, loop_state)

    def decode_dense(self):
        emit_ta, final_state, final_loop_state = tf.nn.raw_rnn(
            self.cell, self.loop_fn, scope=self.scope)
        cand_symbols, cand_logprobs, beam_symbols, beam_logprobs = final_loop_state
        return cand_symbols, cand_logprobs


def beam_decoder(
        cell,
        beam_size,
        stop_token,
        initial_state,
        initial_input,
        max_len=100,
        scope=None
):
    """Beam search decoder

    Args:
        cell: tf.nn.rnn_cell.RNNCell defining the cell to use
        beam_size: the beam size for this search
        stop_token: the index of the symbol used to indicate the end of the
            output
        initial_state: initial cell state for the decoder
        initial_input: initial input into the decoder (typically the embedding
            of a START token)
        max_len: (default 100) maximum length after which to abort beam search.
            This provides an alternative stopping criterion.
        scope: VariableScope for the created subgraph; defaults to "RNN".

    Returns:
        A tuple of the form (decoded, log_probabilities) where:
        decoded: A SparseTensor (or dense Tensor if output_dense=True), of
            underlying shape [batch_size, ?] that contains the decoded sequence
            for each batch element
        log_probability: a [batch_size] tensor containing sequence
            log-probabilities
    """
    with tf.variable_scope(scope or "RNN") as varscope:
        helper = BeamSearchHelper(
            cell=cell,
            beam_size=beam_size,
            stop_token=stop_token,
            initial_state=initial_state,
            initial_input=initial_input,
            max_len=max_len,
            scope=varscope
        )

        return helper.decode_dense()
