
from typing import List, Tuple

import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq

from code.model.abstract.model import Model, LossesType
from code.dataset.abstract.text_dataset import TextDataset
from code.tf_operator import \
    cross_entropy_direct, cross_entropy_summary, \
    attention_supervised_translator, attention_unsupervised_translator


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

    def loss_model(self,
                   source_all: tf.Tensor, target_all: tf.Tensor,
                   length: tf.Tensor,
                   reuse: bool=False) -> Tuple[tf.Tensor, LossesType]:
        source = tf.cast(source_all, tf.int32)
        target = tf.cast(target_all, tf.int32)
        max_length = tf.shape(source)[1]

        logits, _ = attention_supervised_translator(
            source, target, length,
            latent_dim=self.latent_dim,
            voca_size=self.dataset.vocabulary_size,
            num_blocks=self.num_blocks,
            max_length=max_length,
            container=self.embeddings,
            labels=self.dataset.labels,
            name="attention-model",
            reuse=reuse
        )

        loss = cross_entropy_direct(logits, target, name='supervised-x2y')
        loss = cross_entropy_summary(loss, name="supervised-x2y")

        return (loss, [('/cpu:0', loss)])

    def greedy_inference_model(self,
                               source: tf.Tensor, length: tf.Tensor,
                               reuse: bool=False) -> tf.Tensor:
        source = tf.cast(source, tf.int32)
        max_length = tf.shape(source)[1]

        _, labels = attention_unsupervised_translator(
            source, length,
            latent_dim=self.latent_dim,
            voca_size=self.dataset.vocabulary_size,
            num_blocks=self.num_blocks,
            max_length=max_length,
            container=self.embeddings,
            labels=self.dataset.labels,
            name="attention-model",
            reuse=reuse
        )

        return labels
