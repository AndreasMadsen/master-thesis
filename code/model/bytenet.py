
from typing import List

import tensorflow as tf
import sugartensor as stf

from code.model.abstract.model import Model
from code.dataset.abstract.text_dataset import TextDataset
from code.tf_operator import \
    bytenet_supervised_translator, \
    bytenet_unsupervised_translator, \
    bytenet_sampling_translator


class ByteNet(Model):
    latent_dim: int
    num_blocks: int

    def __init__(self, dataset: TextDataset,
                 latent_dim: int=400, num_blocks: int=3,
                 save_dir: str='asset/bytenet',
                 **kwargs) -> None:
        super().__init__(dataset, save_dir=save_dir, **kwargs)

        self.latent_dim = latent_dim
        self.num_blocks = num_blocks

    def _build_train_model(self,
                           x: tf.Tensor, y: tf.Tensor,
                           reuse: bool=False) -> tf.Tensor:
        logits, lables = bytenet_supervised_translator(
            x, y,
            voca_size=self.dataset.vocabulary_size,
            latent_dim=self.latent_dim,
            name="bytenet-model",
            reuse=reuse
        )
        return logits

    def _build_test_model(self,
                          x: tf.Tensor,
                          reuse: bool=False) -> tf.Tensor:
        logits, labels = bytenet_unsupervised_translator(
            x,
            voca_size=self.dataset.vocabulary_size,
            latent_dim=self.latent_dim,
            name="bytenet-model",
            reuse=reuse
        )
        return labels

    def _build_sample_model(self,
                            x: tf.Tensor,
                            samples=1,
                            reuse: bool=False) -> tf.Tensor:
        logits, labels = bytenet_sampling_translator(
            x,
            samples=samples,
            voca_size=self.dataset.vocabulary_size,
            latent_dim=self.latent_dim,
            name="bytenet-model",
            reuse=reuse
        )
        return labels

    def _model_loss(self) -> tf.Tensor:
        with tf.name_scope(None, "preprocessing",
                           values=[self.dataset.source, self.dataset.target]):
            # get source and target tensors
            x = tf.cast(self.dataset.source, tf.int32)
            y = tf.cast(self.dataset.target, tf.int32)

        logits = self._build_train_model(x, y)

        with tf.name_scope(None, "optimization", values=[logits, y]):
            # cross entropy loss with logit and mask
            loss = logits.sg_ce(target=y, mask=True)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=0)

        return loss

    def predict(self, sources: List[str], reuse: bool=False) -> List[str]:
        sources = self.dataset.encode_as_batch(sources)

        # build model
        x = stf.placeholder(dtype=stf.int32, shape=sources.shape)
        label = self._build_test_model(x, reuse=reuse)

        # run graph for translating
        with tf.Session() as sess:
            # init session vars
            stf.sg_init(sess)

            # restore parameters
            stf.sg_restore(sess, self._latest_checkpoint())

            pred = sess.run(label, {x: sources})

        return self.dataset.decode_as_batch(pred)

    def sample(self, sources: List[str], samples=10,
               reuse: bool=False) -> List[str]:
        sources = self.dataset.encode_as_batch(sources)

        # build model
        x = stf.placeholder(dtype=stf.int32, shape=sources.shape)
        label = self._build_sample_model(x, samples=samples, reuse=reuse)

        # run graph for translating
        with tf.Session() as sess:
            # init session vars
            stf.sg_init(sess)

            # restore parameters
            stf.sg_restore(sess, self._latest_checkpoint())

            pred = sess.run(label, {x: sources})

        # reshape to (batch * samples) and back to (batch, samples)
        batch_size = pred.shape[0]
        pred = pred.reshape([-1, pred.shape[-1]])
        texts = self.dataset.decode_as_batch(pred)
        return [
            texts[batch_i * samples:(batch_i + 1) * samples]
            for batch_i
            in range(batch_size)
        ]
