
from typing import List, Tuple

import tensorflow as tf
import sugartensor as stf

from code.model.abstract.model import Model, LossesType
from code.dataset.abstract.text_dataset import TextDataset
from code.tf_operator import \
    cross_entropy_direct, \
    bytenet_supervised_translator, \
    bytenet_unsupervised_translator, \
    bytenet_sampling_translator, \
    tower_scope, mean_n, \
    batch_beam_gather, \
    cross_entropy_summary

default_parameters = {
    'v1': {
        'encoder_size': 5,
        'encoder_normalization': 'bn',
        'decoder_normalization': 'ln',
        'num_blocks': 3,
        'latent_dim': 400,
        'act': 'relu',
        'block_type': 'bytenet'
    },
    'v1-nonorm': {
        'encoder_size': 5,
        'encoder_normalization': 'none',
        'decoder_normalization': 'none',
        'num_blocks': 3,
        'latent_dim': 400,
        'act': 'relu',
        'block_type': 'bytenet'
    },
    'v1-selu': {
        'encoder_size': 5,
        'encoder_normalization': 'none',
        'decoder_normalization': 'none',
        'num_blocks': 3,
        'latent_dim': 400,
        'act': 'selu',
        'block_type': 'bytenet'
    },
    'v1-small': {
        'encoder_size': 5,
        'encoder_normalization': 'bn',
        'decoder_normalization': 'ln',
        'num_blocks': 3,
        'latent_dim': 200,
        'act': 'relu',
        'block_type': 'small'
    },
    'v1-small-nonorm': {
        'encoder_size': 5,
        'encoder_normalization': 'none',
        'decoder_normalization': 'none',
        'num_blocks': 3,
        'latent_dim': 200,
        'act': 'relu',
        'block_type': 'small'
    },
    'v1-small-selu': {
        'encoder_size': 5,
        'encoder_normalization': 'none',
        'decoder_normalization': 'none',
        'num_blocks': 3,
        'latent_dim': 200,
        'act': 'selu',
        'block_type': 'small'
    },
    'v2': {
        'encoder_size': 3,
        'encoder_normalization': 'ln',
        'decoder_normalization': 'ln',
        'num_blocks': 6,
        'latent_dim': 800,
        'act': 'relu',
        'block_type': 'bytenet'
    }
}


class ByteNet(Model):
    _gpus: int

    def __init__(self, dataset: TextDataset,
                 latent_dim: int=None, num_blocks: int=None,
                 save_dir: str='asset/bytenet',
                 version='v1',
                 gpus=1,
                 **kwargs) -> None:
        super().__init__(dataset, save_dir=save_dir, **kwargs)

        self._gpus = gpus

        self._parameters = default_parameters[version].copy()
        self._parameters['voca_size'] = self.dataset.vocabulary_size
        if latent_dim is not None:
            self._parameters['latent_dim'] = latent_dim
        if num_blocks is not None:
            self._parameters['num_blocks'] = num_blocks

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
        else:
            source_split = [source_all]
            target_split = [target_all]

        losses = []

        for (index, device), source, target in zip(
                tower_scope(range(self._gpus), reuse=reuse),
                source_split, target_split
        ):
            x = tf.cast(source, tf.int32)
            y = tf.cast(target, tf.int32)

            logits, _ = bytenet_supervised_translator(
                x, y,
                **self._parameters,
                container=self.embeddings,
                labels=self.dataset.labels,
                name="bytenet-model"
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

    def sample_inference_model(self,
                               source: tf.Tensor, length: tf.Tensor,
                               samples=1,
                               reuse: bool=False) -> tf.Tensor:
        x = tf.cast(source, tf.int32)

        logprops, labels = bytenet_sampling_translator(
            x,
            beam_size=samples,
            **self._parameters,
            name="bytenet-model",
            reuse=reuse
        )

        # check if <eos> exists in each sequence
        # eos_found.shape = (batch, beam)
        eos_found = tf.reduce_any(tf.equal(labels, 1), axis=2)
        # set properbility to something very small if <eos> was not found
        # log(epsilon) = -1e9
        log_eps = tf.constant(-1e9, dtype=logprops.dtype)
        logprops = tf.where(eos_found,
                            logprops,
                            tf.fill(tf.shape(logprops), log_eps))

        # sort by logprops
        _, indices = tf.nn.top_k(logprops, k=samples, sorted=True)
        labels = batch_beam_gather(labels, indices)

        return tf.cast(labels, source.dtype)

    def greedy_inference_model(self,
                               source: tf.Tensor, length: tf.Tensor,
                               reuse: bool=False) -> tf.Tensor:
        x = tf.cast(source, tf.int32)

        _, labels = bytenet_unsupervised_translator(
            x,
            **self._parameters,
            name="bytenet-model",
            reuse=reuse
        )

        return tf.cast(labels, source.dtype)
