
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

        self._vocab_size = self.dataset.vocabulary_size
        self._batch_size = self.dataset.batch_size
        self._output_sos_id = 0
        self._output_eos_id = 1

    def _build_model(self, source, length,
                     helper_build_fn, decoder_maxiters=None):
        batch_size = source.get_shape().as_list()[0]
        if batch_size is None:
            batch_size = tf.shape(source)[0]

        with tf.name_scope('encoder'):
            # embed input_data into a one-hot representation
            inputs = tf.one_hot(source, self._vocab_size)

            fw_cell = rnn.MultiRNNCell([
                rnn.BasicRNNCell(self.latent_dim) for i in range(self.num_blocks)
            ], state_is_tuple=True)
            bw_cell = rnn.MultiRNNCell([
                rnn.BasicRNNCell(self.latent_dim) for i in range(self.num_blocks)
            ], state_is_tuple=True)

            enc_out, _ = tf.nn.bidirectional_dynamic_rnn(
                fw_cell, bw_cell,
                inputs,
                sequence_length=length,
                initial_state_fw=fw_cell.zero_state(batch_size, tf.float32),
                initial_state_bw=bw_cell.zero_state(batch_size, tf.float32)
            )
            enc_out = tf.concat(enc_out, 2)

        with tf.name_scope('decoder'):
            dec_attn = seq2seq.DynamicAttentionWrapper(
                cell=rnn.GRUCell(self.latent_dim * 2),
                attention_mechanism=seq2seq.LuongAttention(self.latent_dim * 2, enc_out, length),
                attention_size=self.latent_dim * 2
            )

            dec_network = rnn.MultiRNNCell([
                rnn.GRUCell(self.latent_dim * 2),
                dec_attn,
                rnn.GRUCell(self._vocab_size)
            ], state_is_tuple=True)

            decoder = seq2seq.BasicDecoder(
                dec_network, helper_build_fn(),
                initial_state=dec_network.zero_state(batch_size, tf.float32)
            )

            dec_outputs, _ = seq2seq.dynamic_decode(
                decoder,
                maximum_iterations=decoder_maxiters,
                impute_finished=True
            )

        return dec_outputs.rnn_output, dec_outputs.sample_id

    def loss_model(self,
                   source_all: tf.Tensor, target_all: tf.Tensor,
                   reuse: bool=False) -> Tuple[tf.Tensor, LossesType]:
        source = tf.cast(self.dataset.source, tf.int32)
        target = tf.cast(self.dataset.target, tf.int32)
        length = self.dataset.length
        max_length = tf.shape(source)[1]

        def train_helper():
            batch_size = int(source.get_shape()[0])
            start_ids = tf.fill([batch_size, 1], self._output_sos_id)
            decoder_input_ids = tf.concat([start_ids, target], 1)
            decoder_inputs = tf.one_hot(decoder_input_ids, self._vocab_size)
            return seq2seq.TrainingHelper(decoder_inputs, length)

        logits, labels = self._build_model(source, length, train_helper,
                                           decoder_maxiters=max_length)
        logits = tf.slice(logits, [0, 0, 0], [-1, max_length, -1])

        loss = cross_entropy_direct(logits, target, name='attention')
        loss = cross_entropy_summary(loss, name="supervised-x2y")

        return (loss, [('/cpu:0', loss)])

    def greedy_inference_model(self,
                               source: tf.Tensor,
                               reuse: bool=False) -> tf.Tensor:
        source = tf.cast(source, tf.int32)
        # length # should be passed
        max_length = tf.shape(source)[1]

        def infer_helper():
            batch_size = tf.shape(source)[0]
            return seq2seq.GreedyEmbeddingHelper(
                    lambda ids: tf.one_hot(ids, self._vocab_size),
                    start_tokens=tf.fill([batch_size], self._output_sos_id),
                    end_token=self._output_eos_id)

        logits, labels = self._build_model(
            source, None, infer_helper, decoder_maxiters=max_length
        )

        return labels
