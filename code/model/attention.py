
from typing import List, Tuple

import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq

from code.model.abstract.model import Model, LossesType
from code.dataset.abstract.text_dataset import TextDataset
from code.tf_operator import \
    cross_entropy_direct, cross_entropy_summary


class Attention(Model):
    def __init__(self, dataset: TextDataset,
                 latent_dim: int=20, num_blocks: int=3,
                 save_dir: str='asset/attention',
                 gpus=1,
                 **kwargs) -> None:
        super().__init__(dataset, save_dir=save_dir, **kwargs)

        self.dataset = dataset
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim

        self._out_help = True
        self._vocab_size = self.dataset.vocabulary_size
        self._batch_size = self.dataset.batch_size
        self._output_sos_id = 0
        self._output_eos_id = 1

    def _build_model(self, batch_size, helper_build_fn, decoder_maxiters=None):
        # embed input_data into a one-hot representation
        inputs = tf.one_hot(self.input_data, self._vocab_size)
        inputs_len = self.input_lengths

        with tf.name_scope('encoder'):
            fw_cell = rnn.MultiRNNCell([
                rnn.BasicRNNCell(self.latent_dim) for i in range(self.num_blocks)
            ], state_is_tuple=True)
            bw_cell = rnn.MultiRNNCell([
                rnn.BasicRNNCell(self.latent_dim) for i in range(self.num_blocks)
            ], state_is_tuple=True)
            fw_cell_zero = fw_cell.zero_state(batch_size, tf.float32)
            bw_cell_zero = bw_cell.zero_state(batch_size, tf.float32)

            enc_out, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell, inputs,
                sequence_length=inputs_len,
                initial_state_fw=fw_cell_zero,
                initial_state_bw=bw_cell_zero)

        with tf.name_scope('decoder'):
            dec_cell_in = rnn.GRUCell(self.latent_dim * 2)
            attn_values = tf.concat(enc_out, 2)
            attn_mech = seq2seq.LuongAttention(self.latent_dim * 2, attn_values, inputs_len)
            dec_cell_attn = rnn.GRUCell(self.latent_dim * 2)
            dec_cell_attn = seq2seq.DynamicAttentionWrapper(dec_cell_attn, attn_mech, self.latent_dim * 2)
            dec_cell_out = rnn.GRUCell(self._vocab_size)
            dec_cell = rnn.MultiRNNCell([dec_cell_in, dec_cell_attn, dec_cell_out],
                                        state_is_tuple=True)

            dec = seq2seq.BasicDecoder(dec_cell, helper_build_fn(),
                                       dec_cell.zero_state(batch_size, tf.float32))

            dec_out, _ = seq2seq.dynamic_decode(dec, output_time_major=False,
                    maximum_iterations=decoder_maxiters, impute_finished=True)

        self.outputs = dec_out.rnn_output
        self.output_ids = dec_out.sample_id

    def _output_onehot(self, ids):
        return tf.one_hot(ids, self._vocab_size)

    def loss_model(self,
                   source_all: tf.Tensor, target_all: tf.Tensor,
                   reuse: bool=False) -> Tuple[tf.Tensor, LossesType]:
        self.input_data = tf.cast(self.dataset.source, tf.int32)
        self.input_lengths = self.dataset.length
        self.output_data = tf.cast(self.dataset.target, tf.int32)
        self.output_lengths = self.dataset.length

        output_data_maxlen = tf.shape(self.output_data)[1]

        def infer_helper():
            return seq2seq.GreedyEmbeddingHelper(
                    self._output_onehot,
                    start_tokens=tf.fill([self._batch_size], self._output_sos_id),
                    end_token=self._output_eos_id)

        def train_helper():
            start_ids = tf.fill([self._batch_size, 1], self._output_sos_id)
            decoder_input_ids = tf.concat([start_ids, self.output_data], 1)
            decoder_inputs = self._output_onehot(decoder_input_ids)
            return seq2seq.TrainingHelper(decoder_inputs, self.output_lengths)

        helper = train_helper if self._out_help else infer_helper

        self._build_model(self._batch_size, helper, decoder_maxiters=output_data_maxlen)

        output_maxlen = tf.minimum(tf.shape(self.outputs)[1], output_data_maxlen)
        out_data_slice = tf.slice(self.output_data, [0, 0], [-1, output_maxlen])
        out_logits_slice = tf.slice(self.outputs, [0, 0, 0], [-1, output_maxlen, -1])

        loss = cross_entropy_direct(out_logits_slice, out_data_slice,
                                    name='attention')
        loss = cross_entropy_summary(loss,
                                     name="supervised-x2y")

        return (loss, [('/cpu:0', loss)])

    def greedy_inference_model(self,
                               source: tf.Tensor,
                               reuse: bool=False) -> tf.Tensor:
        self.input_data = tf.cast(source, tf.int32)
        self.input_lengths = None

        batch_size = tf.shape(source)[0]

        def infer_helper():
            return seq2seq.GreedyEmbeddingHelper(
                    self._output_onehot,
                    start_tokens=tf.fill([batch_size], self._output_sos_id),
                    end_token=self._output_eos_id)

        self._build_model(
            batch_size, infer_helper, decoder_maxiters=128
        )

        return self.output_ids
