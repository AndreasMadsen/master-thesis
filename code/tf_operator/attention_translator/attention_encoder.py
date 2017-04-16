
import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq


def attention_encoder(x, length,
                      num_blocks=3,
                      name=None, reuse=None):
    with tf.variable_scope(name, "attention-encoder", values=[x, length],
                           reuse=reuse):
        # get shapes
        batch_size = x.get_shape().as_list()[0]
        if batch_size is None:
            batch_size = tf.shape(x)[0]

        dims = int(x.get_shape()[-1])

        # encode data
        fw_cell = rnn.MultiRNNCell([
            rnn.BasicRNNCell(dims) for i in range(num_blocks)
        ], state_is_tuple=True)
        bw_cell = rnn.MultiRNNCell([
            rnn.BasicRNNCell(dims) for i in range(num_blocks)
        ], state_is_tuple=True)

        enc_out, _ = tf.nn.bidirectional_dynamic_rnn(
            fw_cell, bw_cell,
            x,
            sequence_length=length,
            initial_state_fw=fw_cell.zero_state(batch_size, tf.float32),
            initial_state_bw=bw_cell.zero_state(batch_size, tf.float32)
        )
        enc_out = tf.concat(enc_out, 2)

        return enc_out
