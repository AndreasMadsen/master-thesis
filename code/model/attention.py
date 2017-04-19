
from typing import List, Tuple

import tensorflow as tf
from tensorflow.contrib import rnn, seq2seq

from code.model.abstract.model import Model, LossesType
from code.dataset.abstract.text_dataset import TextDataset
from code.tf_operator import \
    cross_entropy_direct, cross_entropy_summary, \
    attention_supervised_translator, attention_unsupervised_translator, \
    tower_scope, mean_n


class Attention(Model):
    def __init__(self, dataset: TextDataset,
                 latent_dim: int=20, num_blocks: int=3,
                 save_dir: str='asset/attention',
                 gpus=1,
                 **kwargs) -> None:
        super().__init__(dataset, save_dir=save_dir, **kwargs)

        self._gpus = gpus
        self.dataset = dataset
        self.num_blocks = num_blocks
        self.latent_dim = latent_dim

    def loss_model(self,
                   source_all: tf.Tensor, target_all: tf.Tensor,
                   length_all: tf.Tensor,
                   reuse: bool=False) -> Tuple[tf.Tensor, LossesType]:
        # putting the split and join on the cpu is extreamly important for
        # minimizing the syncronization time.
        if self._gpus > 1:
            with tf.device('/cpu:0'):
                source_split = tf.split(source_all, self._gpus, axis=0)
                target_split = tf.split(target_all, self._gpus, axis=0)
                length_split = tf.split(length_all, self._gpus, axis=0)
        else:
            source_split = [source_all]
            target_split = [target_all]
            length_split = [length_all]

        losses = []

        for (index, device), source, target, length in zip(
                tower_scope(range(self._gpus), reuse=reuse),
                source_split, target_split, length_split
        ):
            x = tf.cast(source, tf.int32)
            y = tf.cast(target, tf.int32)

            logits, _ = attention_supervised_translator(
                x, y, length,
                latent_dim=self.latent_dim,
                voca_size=self.dataset.vocabulary_size,
                num_blocks=self.num_blocks,
                max_length=tf.shape(source)[1],
                container=self.embeddings,
                labels=self.dataset.labels,
                name="attention-model",
                reuse=reuse
            )

            losses.append(
                (device, cross_entropy_direct(logits, y, "supervised-x2y"))
            )

        # join the losses
        if self._gpus > 1:
            with tf.device('/cpu:0'):
                total_loss = mean_n([loss for _, loss in losses])
                total_loss = cross_entropy_summary(total_loss,
                                                   name="supervised-x2y",
                                                   reuse=reuse)
        else:
            _, total_loss = losses[0]
            total_loss = cross_entropy_summary(total_loss,
                                               name="supervised-x2y",
                                               reuse=reuse)

        return (total_loss, losses)

    def greedy_inference_model(self,
                               source: tf.Tensor, length: tf.Tensor,
                               reuse: bool=False) -> tf.Tensor:
        x = tf.cast(source, tf.int32)
        max_length = tf.shape(source)[1]

        _, labels = attention_unsupervised_translator(
            x, length,
            latent_dim=self.latent_dim,
            voca_size=self.dataset.vocabulary_size,
            num_blocks=self.num_blocks,
            max_length=max_length,
            container=self.embeddings,
            labels=self.dataset.labels,
            name="attention-model",
            reuse=reuse
        )

        return tf.cast(labels, source.dtype)
