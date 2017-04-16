
import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq


def attention_decoder(enc, length, state_transfer_helper,
                      voca_size=20, max_length=None,
                      name=None, reuse=None):
    with tf.variable_scope(name, "attention-decoder", values=[enc, length],
                           reuse=reuse):
        # get shapes
        batch_size = enc.get_shape().as_list()[0]
        if batch_size is None:
            batch_size = tf.shape(enc)[0]

        dims = int(enc.get_shape()[-1])

        # decoder
        dec_attn = seq2seq.DynamicAttentionWrapper(
            cell=rnn.GRUCell(dims),
            attention_mechanism=seq2seq.LuongAttention(dims, enc, length),
            attention_size=dims
        )

        dec_network = rnn.MultiRNNCell([
            rnn.GRUCell(dims),
            dec_attn,
            rnn.GRUCell(voca_size)
        ], state_is_tuple=True)

        decoder = seq2seq.BasicDecoder(
            dec_network, state_transfer_helper(),
            initial_state=dec_network.zero_state(batch_size, tf.float32)
        )

        dec_outputs, _ = seq2seq.dynamic_decode(
            decoder,
            maximum_iterations=max_length,
            impute_finished=True
        )

        return dec_outputs.rnn_output, dec_outputs.sample_id
