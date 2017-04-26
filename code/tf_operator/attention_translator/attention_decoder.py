
import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq


def dynamic_time_pad(tensor, max_length):
    shape = tensor.get_shape().as_list()
    shape_op = tf.shape(tensor)

    pad_size = max_length - shape_op[1]

    if len(shape) == 3:
        tensor = tf.concat([
            tensor,
            tf.zeros([shape_op[0], pad_size, shape[2]], dtype=tensor.dtype)
        ], 1)
    elif len(shape) == 2:
        tensor = tf.concat([
            tensor,
            tf.zeros([shape_op[0], pad_size], dtype=tensor.dtype)
        ], 1)
    else:
        raise NotImplemented(f'tensor with {len(shape)} dimentions.')

    return tensor


def attention_decoder(enc, length, state_transfer_helper,
                      voca_size=20, max_length=None,
                      name=None, reuse=None):
    with tf.variable_scope(name, "attention-decoder", values=[enc, length],
                           reuse=reuse) as scope:
        # get shapes
        batch_size = enc.get_shape().as_list()[0]
        if batch_size is None:
            batch_size = tf.shape(enc)[0]

        dims = int(enc.get_shape()[-1])

        # decoder
        dec_attn = seq2seq.DynamicAttentionWrapper(
            cell=rnn.GRUCell(dims, reuse=scope.reuse),
            attention_mechanism=seq2seq.LuongAttention(dims, enc, length),
            attention_size=dims
        )

        dec_network = rnn.MultiRNNCell([
            rnn.GRUCell(dims, reuse=scope.reuse),
            dec_attn,
            rnn.GRUCell(voca_size, reuse=scope.reuse)
        ], state_is_tuple=True)

        decoder = seq2seq.BasicDecoder(
            dec_network, state_transfer_helper(),
            initial_state=dec_network.zero_state(batch_size, tf.float32)
        )

        dec_outputs, _ = seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=max_length,
            impute_finished=False
        )

        logits = dec_outputs.rnn_output
        labels = dec_outputs.sample_id

        # pad logits and labels
        if max_length is not None:
            logits = dynamic_time_pad(logits, max_length)
            labels = dynamic_time_pad(labels, max_length)

        return logits, labels
