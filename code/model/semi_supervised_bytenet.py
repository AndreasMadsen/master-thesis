
from typing import List

import tensorflow as tf
import sugartensor as stf

from code.model.abstract.model import Model
from code.dataset.abstract.text_dataset import TextDataset
from code.tf_operator import \
    batch_repeat, batch_repeat_pack, batch_repeat_unpack, \
    select_dim_value, seq_prop, \
    bytenet_supervised_translator, \
    bytenet_unsupervised_translator, \
    bytenet_sampling_translator


class SemiSupervisedByteNet(Model):
    latent_dim: int
    num_blocks: int
    samples: int
    dataset_x: TextDataset
    dataset_y: TextDataset

    def __init__(self,
                 dataset_x2y: TextDataset,
                 dataset_x: TextDataset=None, dataset_x_loss_factor=0.1,
                 dataset_y: TextDataset=None, dataset_y_loss_factor=0.1,
                 latent_dim: int=400, num_blocks: int=3, samples: int=10,
                 save_dir: str='asset/semi-bytenet',
                 **kwargs) -> None:
        super().__init__(dataset_x2y, **kwargs)

        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.samples = samples

        self.dataset_x = dataset_x
        self.dataset_x_loss_factor = dataset_x_loss_factor
        self.dataset_y = dataset_y
        self.dataset_y_loss_factor = dataset_y_loss_factor

    def _build_supervised_model(self,
                                x: tf.Tensor, y: tf.Tensor,
                                order: str=None,
                                reuse: bool=False) -> tf.Tensor:
        name, name_scope = (None, None)
        if order is not None:
            name = f'bytenet-{order}'
            name_scope = f'loss-supervised-bytenet-{order}'

        with tf.name_scope(name_scope, 'loss-supervised', values=[x, y]):
            logits, lables = bytenet_supervised_translator(
                x, y,
                voca_size=self.dataset.vocabulary_size,
                latent_dim=self.latent_dim,
                num_blocks=self.num_blocks,
                name=name,
                reuse=reuse
            )

            # cross entropy loss with logit and mask
            loss = logits.sg_ce(target=y, mask=True)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1), axis=0)

            return tf.check_numerics(loss, f'loss-supervised-{order}')

    def _build_unsupervised_model(self,
                                  x: tf.Tensor,
                                  order: str=None,
                                  reuse: bool=False) -> tf.Tensor:
        name_forward, name_backward, name_scope = (None, None, None)
        if order is not None:
            name_forward = f'bytenet-{order}'
            name_backward = f'bytenet-{order[::-1]}'
            name_scope = f'loss-unsupervised-bytenet-{order}-{order[::-1]}'

        with tf.name_scope(name_scope, 'loss-unsupervised', values=[x]):
            # x.shape = (batch, time)
            # x_repeat.shape = (batch, samples, time)
            x_repeat = batch_repeat(x, repeats=self.samples)

            # sample_logits.shape = (batch, samples, times, vocabulary)
            # sample_labels.shape = (batch, samples, times)
            sample_logits, sample_labels = bytenet_sampling_translator(
                x,
                voca_size=self.dataset.vocabulary_size,
                latent_dim=self.latent_dim,
                num_blocks=self.num_blocks,
                samples=self.samples,
                name=name_forward,
                reuse=reuse
            )
            # sample_props.shape = (batch, samples, times)
            sample_props = tf.nn.softmax(sample_logits)
            sample_props = select_dim_value(sample_props, sample_labels)

            # logits.shape = (batch * samples, times, vocabulary)
            # labels.shape = (batch * samples, times)
            logits, labels = bytenet_supervised_translator(
                batch_repeat_pack(sample_labels), batch_repeat_pack(x_repeat),
                voca_size=self.dataset.vocabulary_size,
                latent_dim=self.latent_dim,
                num_blocks=self.num_blocks,
                name=name_backward,
                reuse=reuse
            )

            # logits.shape = (batch, samples, times, vocabulary)
            logits = batch_repeat_unpack(logits, repeats=self.samples)

            # props.shape = (batch, samples, times, vocabulary)
            props = tf.nn.softmax(logits)
            # props.shape = (batch, samples, times)
            props = select_dim_value(props, x_repeat)

            # sample_seq_props.shape = (batch, samples)
            sample_seq_props = seq_prop(sample_props, mask=sample_labels,
                                        axis=2)
            # seq_props.shape = (batch, samples)
            seq_props = seq_prop(props, mask=x_repeat, axis=2)

            # marginal_props.shape (batch, )
            marginal_props = tf.reduce_sum(sample_seq_props * seq_props,
                                           axis=1)

            # cross entropy loss (scalar)
            loss = -tf.reduce_mean(tf.log(marginal_props + 1e-9), axis=0)

            return tf.check_numerics(loss, f'loss-unsupervised-{order}')

    def _build_test_model(self,
                          x: tf.Tensor,
                          order='x2y',
                          reuse: bool=False) -> tf.Tensor:
        logits, labels = bytenet_unsupervised_translator(
            x,
            voca_size=self.dataset.vocabulary_size,
            latent_dim=self.latent_dim,
            num_blocks=self.num_blocks,
            name=f'bytenet-{order}',
            reuse=reuse
        )
        return labels

    def _model_loss(self) -> tf.Tensor:
        loss = 0

        with tf.name_scope(None, "supervised",
                           values=[self.dataset.source, self.dataset.target]):
            # get source and target tensors
            x = tf.cast(self.dataset.source, tf.int32)
            y = tf.cast(self.dataset.target, tf.int32)

            loss += self._build_supervised_model(x, y, order='x2y')
            loss += self._build_supervised_model(y, x, order='y2x')

        if self.dataset_x is not None:
            with tf.name_scope(None, "unsupervised-x",
                               values=[self.dataset_x.source]):
                x = tf.cast(self.dataset_x.source, tf.int32)

                loss_x = self._build_unsupervised_model(
                    x, order='x2y', reuse=True
                )
                loss += self.dataset_x_loss_factor * loss_x

        if self.dataset_y is not None:
            with tf.name_scope(None, "unsupervised-y",
                               values=[self.dataset_y.source]):
                y = tf.cast(self.dataset_y.source, tf.int32)

                loss_y = self._build_unsupervised_model(
                    y, order='y2x', reuse=True
                )
                loss += self.dataset_y_loss_factor * loss_y

        return loss

    def predict(self, sources: List[str], order='x2y',
                reuse: bool=False) -> List[str]:
        sources = self.dataset.encode_as_batch(sources)

        # get source and target tensors
        x = stf.placeholder(dtype=stf.int32, shape=sources.shape)

        label = self._build_test_model(x, order=order, reuse=reuse)

        # run graph for translating
        with tf.Session() as sess:
            # init session vars
            stf.sg_init(sess)

            # restore parameters
            stf.sg_restore(sess, self._latest_checkpoint())

            pred = sess.run(label, {x: sources})

        return self.dataset.decode_as_batch(pred)
