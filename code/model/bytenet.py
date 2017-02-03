
from typing import List

import tensorflow as tf
import sugartensor as stf
import numpy as np

from code.model.abstract.model import Model
from code.dataset.abstract.text_dataset import TextDataset
from code.tf_operator import \
    bytenet_supervised_translator, \
    bytenet_unsupervised_translator


class ByteNet(Model):
    latent_dim: int
    num_blocks: int
    _save_dir: str

    def __init__(self, dataset: TextDataset,
                 latent_dim: int=400, num_blocks: int=3,
                 save_dir='asset/bytenet') -> None:
        super().__init__(dataset)

        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self._save_dir = save_dir

    def _build_train_model(self,
                           x: tf.Tensor, y: tf.Tensor,
                           reuse=False) -> tf.Tensor:
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
                          reuse=False) -> tf.Tensor:
        logits, labels = bytenet_unsupervised_translator(
            x,
            voca_size=self.dataset.vocabulary_size,
            latent_dim=self.latent_dim,
            name="bytenet-model",
            reuse=reuse
        )
        return labels

    def train(self, max_ep=20, lr=0.0001, **kwargs):
        with tf.name_scope(None, "preprocessing",
                           values=[self.dataset.source, self.dataset.target]):
            # get source and target tensors
            x = tf.cast(self.dataset.source, tf.int32)
            y = tf.cast(self.dataset.target, tf.int32)

        dec = self._build_train_model(x, y)

        with tf.name_scope(None, "optimization", values=[dec, y]):
            # cross entropy loss with logit and mask
            loss = dec.sg_ce(target=y, mask=True)

        # train
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=sess_config) as sess:
            stf.sg_train(log_interval=30,
                         lr=lr,
                         loss=loss,
                         ep_size=self.dataset.num_batch,
                         max_ep=max_ep,
                         early_stop=False,
                         save_dir=self._save_dir,
                         sess=sess,
                         **kwargs)

    def predict(self, sources, reuse=False) -> List[str]:
        sources = self.dataset.encode_as_batch(sources)
        predict_shape = (sources.shape[0], self.dataset.effective_max_length)

        # get source and target tensors
        x = stf.placeholder(dtype=stf.int32, shape=sources.shape)

        label = self._build_test_model(x, reuse=reuse)

        # run graph for translating
        with tf.Session() as sess:
            # init session vars
            stf.sg_init(sess)

            # restore parameters
            stf.sg_restore(sess, tf.train.latest_checkpoint(self._save_dir))

            pred = sess.run(label, {x: sources})

        return self.dataset.decode_as_batch(pred)
