
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
    batch_beam_gather


class ByteNet(Model):
    latent_dim: int
    num_blocks: int
    _gpus: int

    def __init__(self, dataset: TextDataset,
                 latent_dim: int=400, num_blocks: int=3,
                 save_dir: str='asset/bytenet',
                 gpus=1,
                 **kwargs) -> None:
        super().__init__(dataset, save_dir=save_dir, **kwargs)

        self._gpus = gpus
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks

    def loss_model(self,
                   source_all: tf.Tensor, target_all: tf.Tensor,
                   reuse: bool=False) -> Tuple[tf.Tensor, LossesType]:
        # putting the split and join on the cpu is extreamly important for
        # minimizing the syncronization time.
        with tf.device('/cpu:0'):
            source_split = tf.split(source_all, self._gpus, 0)
            target_split = tf.split(target_all, self._gpus, 0)

        losses = []

        for (index, device), source, target in zip(
                tower_scope(range(self._gpus), reuse=reuse),
                source_split, target_split
        ):
            x = tf.cast(source, tf.int32)
            y = tf.cast(target, tf.int32)

            logits, _ = bytenet_supervised_translator(
                x, y,
                voca_size=self.dataset.vocabulary_size,
                latent_dim=self.latent_dim,
                num_blocks=self.num_blocks,
                container=self.embeddings,
                labels=self.dataset.labels,
                name="bytenet-model"
            )

            losses.append(
                (device, cross_entropy_direct(logits, y, "supervised-x2y"))
            )

        # join the losses
        with tf.device('/cpu:0'):
            total_loss = mean_n([loss for _, loss in losses])

        return (total_loss, losses)

    def sample_model(self,
                     source: tf.Tensor,
                     samples=1,
                     reuse: bool=False) -> tf.Tensor:
        x = tf.cast(source, tf.int32)

        logprops, labels = bytenet_sampling_translator(
            x,
            beam_size=samples,
            voca_size=self.dataset.vocabulary_size,
            latent_dim=self.latent_dim,
            num_blocks=self.num_blocks,
            name="bytenet-model",
            reuse=reuse
        )

        # sort by logprops
        _, indices = tf.nn.top_k(logprops, k=samples, sorted=True)
        labels = batch_beam_gather(labels, indices)

        return tf.cast(labels, source.dtype)

    def inference_model(self,
                        source: tf.Tensor,
                        reuse: bool=False) -> tf.Tensor:
        x = tf.cast(source, tf.int32)

        _, labels = bytenet_unsupervised_translator(
            x,
            voca_size=self.dataset.vocabulary_size,
            latent_dim=self.latent_dim,
            num_blocks=self.num_blocks,
            name="bytenet-model",
            reuse=reuse
        )

        return tf.cast(labels, source.dtype)
