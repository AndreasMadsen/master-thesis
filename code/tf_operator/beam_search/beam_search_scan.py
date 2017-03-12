
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from code.tf_operator.select_value.batch_beam_gather import batch_beam_gather
from code.tf_operator.batch_repeat.batch_repeat import batch_repeat
from code.tf_operator.batch_repeat.batch_repeat_pack import \
    batch_repeat_pack
from code.tf_operator.batch_repeat.batch_repeat_unpack import \
    batch_repeat_unpack


def _transpose_to_time(tensor):
    permutation = list(range(len(tensor.get_shape())))
    permutation[0] = 2
    permutation[1] = 0
    permutation[2] = 1
    return tf.transpose(tensor, permutation)


def _transpose_to_batch(tensor):
    permutation = list(range(len(tensor.get_shape())))
    permutation[0] = 1
    permutation[1] = 2
    permutation[2] = 0
    return tf.transpose(tensor, permutation)


def beam_search_scan(scan_func, elems=None, initializer=None,
                     beam_size=2,
                     name=None, **kwargs):
    '''
    scan like API but wraps scan_func with a beam search.

    Arguments:
        scan_func: (state_t, logits_t) = scan_func(
                    (state_tm1, logits_tm1, label_tm1), elems_t
                   )

        elems: The input sequence elements. shape = (batch, time, ...)

        initializer:
            state_tm1: The initial state.
            logits_tm1: The initial logits. shape = (batch, dims)
            label_tm1: The initial labels. shape = (batch, )

    Returns:
        state: The Scan state output, mostly useless.
        logprops: The scan output for the sequence log properbilities.
                  shape = (batch, beam, time)
        labels: The label sequences.
                shape = (batch, beam, time)
    '''
    if elems is None:
        raise ValueError('elems must be specified in beam search scan')
    if not isinstance(initializer, tuple) or len(initializer) != 3:
        raise ValueError('initializer must be tuple(state, logits, labels)')

    # convert input values to tensors
    elems = tf.convert_to_tensor(elems)
    initializer = nest.map_structure(
        lambda item: tf.convert_to_tensor(item),
        initializer
    )

    with tf.name_scope(name, 'beam-search-scan',
                       values=[elems] + nest.flatten(initializer)):
        # get shape parameters
        batch_size_op = tf.shape(elems)[0]
        batch_size_val = elems.get_shape()[0]  # possibly None
        time_size_op = tf.shape(elems)[1]
        time_size_val = elems.get_shape()[1]  # possibly None

        # initialize time communicative normalized logits:
        #   log(P(state|t=0)) = 0, because P(·|t=0) = 1
        logprobs = tf.concat([
            tf.zeros((batch_size_op, 1), dtype=tf.float32),
            tf.fill((batch_size_op, beam_size - 1), -np.inf),
        ], axis=1)
        logprobs.set_shape([batch_size_val, beam_size])
        # initialize ended bool vectors:
        ended = tf.zeros((batch_size_op, beam_size), dtype=tf.bool)
        ended.set_shape([batch_size_val, beam_size])
        # initialize full label sequence container
        labels_full = tf.zeros((batch_size_op, beam_size, time_size_op),
                               dtype=tf.int32)
        labels_full.set_shape([batch_size_val, beam_size, time_size_val])

        # repeat elems and initializer `beam_size` times
        elems = batch_repeat(elems, repeats=beam_size)
        initializer = nest.map_structure(
            lambda item: batch_repeat(item, repeats=beam_size),
            initializer
        )

        # transpose time axis first
        elems = _transpose_to_time(elems)

        (state, logits, labels, logprops, ended, labels_full, time) = tf.scan(
            lambda *args, **kwargs: _scan_wrapper(
                scan_func,
                batch_size_op, batch_size_val, beam_size,
                *args, **kwargs
            ),
            elems=elems,
            initializer=(*initializer, logprobs, ended, labels_full, 0),
            **kwargs
        )

        # transpose batch axis first
        state = nest.map_structure(
            lambda item: _transpose_to_batch(item),
            state
        )
        logprops = _transpose_to_batch(logprops)

        return (state, logprops, labels_full[-1])


def _pack(*args):
    return [nest.map_structure(
        lambda item: batch_repeat_pack(item), arg
    ) for arg in args]


def _unpack(*args, repeats=1):
    return [nest.map_structure(
        lambda item: batch_repeat_unpack(item, repeats=repeats), arg
    ) for arg in args]


def _scan_wrapper(scan_func,
                  batch_size_op, batch_size_val, beam_size,
                  prev_tm1, elems_t):
    (state_tm1, logits_tm1, labels_tm1,
     logprops_tm1, ended_tm1, labels_full_tm1,
     time_tm1) = prev_tm1

    # pack input, call `scan_func`, unpack output
    input_pack = _pack((state_tm1, logits_tm1, labels_tm1), elems_t)
    with tf.name_scope(None, 'beam-search-scan-func', values=input_pack):
        output_pack = scan_func(*input_pack)
    (state_t, logits_t) = _unpack(*output_pack, repeats=beam_size)

    # get vocabulary size
    vocab_size = int(logits_t.get_shape()[-1])

    # calculate logprops_t_current
    # logprops_t_current.shape = (batch, beam, vocab)
    logprops_t_current = tf.nn.log_softmax(logits_t)
    logprops_t_current = tf.reshape(logprops_t_current, [-1, vocab_size])
    logprops_t_fill = tf.tile(
        # create a log(P(·)) vector where P(<null>) = 1
        tf.constant([[0] + [-np.inf] * (vocab_size - 1)], dtype=elems_t.dtype),
        (batch_size_op * beam_size, 1)
    )
    # force P(<null>|y_{t-1} = {<eos>, <null>}) = 1
    logprops_t_current = tf.where(
        tf.reshape(ended_tm1, [-1]),
        logprops_t_fill, logprops_t_current
    )
    logprops_t_current = tf.reshape(logprops_t_current,
                                    [batch_size_op, beam_size, vocab_size])
    logprops_t_current.set_shape([batch_size_val, beam_size, vocab_size])

    # calculate logprops_t
    # log(P(y|T<=t)) = log(P(y|T<t)  * P(y|T=t))
    #                = log(P(y|T<t)) + log(P(y|T=t))
    #                = log(P(y|T<t)) + logsoftmax(logits)
    # logprops_t_full.shape = (batch, beam, vocab)
    logprops_t_full = tf.expand_dims(logprops_tm1, -1) + logprops_t_current

    # select `beam_size`-most likely properbilities from logprops_t_full
    # {logprops_t, logprops_t_indices}.shape = (batch, beam)
    logprops_t_full_compact = tf.reshape(logprops_t_full,
                                         [-1, beam_size * vocab_size])
    logprops_t_full_compact.set_shape([batch_size_val, beam_size * vocab_size])
    logprops_t, logprops_t_indices = tf.nn.top_k(logprops_t_full_compact,
                                                 k=beam_size, sorted=False)

    # convert logprops_t_indices to label_t and beam_t (beam indices)
    # {label_t, beam_t}.shape = (batch, beam)
    labels_t = logprops_t_indices % vocab_size
    beam_t = logprops_t_indices // vocab_size

    # transfer and copy labels_full_t
    time_t = time_tm1 + 1
    labels_full_t = batch_beam_gather(labels_full_tm1, beam_t)
    labels_full_t = tf.concat([
        labels_full_t[:, :, :time_tm1],
        tf.expand_dims(labels_t, -1),
        labels_full_t[:, :, time_t:]
    ], axis=2)
    labels_full_t.set_shape(labels_full_tm1.get_shape())

    # updated eneded state
    ended_t = tf.logical_or(
        # move the ended state in case the beam index changed
        batch_beam_gather(ended_tm1, beam_t),
        tf.equal(labels_t, 1)
    )

    # copy state from selected beam path
    # logits_t.shape = (batch, beam, dims)
    logits_t = batch_beam_gather(logits_t, beam_t)
    state_t = nest.map_structure(
        lambda item: batch_beam_gather(item, beam_t),
        state_t
    )

    return (state_t, logits_t, labels_t,
            logprops_t, ended_t, labels_full_t,
            time_t)
