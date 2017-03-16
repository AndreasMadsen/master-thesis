
from typing import Tuple, List

import tensorflow as tf

from code.model.abstract.model import Model, LossesType
from code.dataset.abstract.text_dataset import TextDataset
from code.tf_operator import \
    batch_repeat, batch_repeat_pack, batch_repeat_unpack, \
    select_dim_value, seq_logprop, \
    cross_entropy_direct, cross_entropy_indirect, \
    bytenet_supervised_translator, \
    bytenet_unsupervised_translator, \
    bytenet_sampling_translator, \
    distributed_tower_optim


class SemiSupervisedByteNet(Model):
    latent_dim: int
    num_blocks: int
    beam_size: int
    dataset_x: TextDataset
    dataset_y: TextDataset

    def __init__(self,
                 dataset_x2y: TextDataset,
                 dataset_x: TextDataset=None, dataset_x_loss_factor=0.1,
                 dataset_y: TextDataset=None, dataset_y_loss_factor=0.1,
                 latent_dim: int=400, num_blocks: int=3, beam_size: int=10,
                 save_dir: str='asset/semi-bytenet',
                 **kwargs) -> None:
        super().__init__(dataset_x2y, save_dir=save_dir, **kwargs)

        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.beam_size = beam_size

        self.dataset_x = dataset_x
        self.dataset_x_loss_factor = dataset_x_loss_factor
        self.dataset_y = dataset_y
        self.dataset_y_loss_factor = dataset_y_loss_factor

    def _build_supervised_model(self,
                                source: tf.Tensor, target: tf.Tensor,
                                order: str=None,
                                reuse: bool=False) -> tf.Tensor:
        # get source and target tensors
        source = tf.cast(source, tf.int32)
        target = tf.cast(target, tf.int32)

        name, name_scope = (None, None)
        if order is not None:
            name = f'bytenet-{order}'
            name_scope = f'supervised-bytenet-{order}'

        with tf.name_scope(name_scope, 'supervised', values=[source, target]):
            logits, lables = bytenet_supervised_translator(
                source, target,
                voca_size=self.dataset.vocabulary_size,
                latent_dim=self.latent_dim,
                num_blocks=self.num_blocks,
                container=self.embeddings,
                labels=self.dataset.labels,
                name=name,
                reuse=reuse
            )

        loss = cross_entropy_direct(logits, target,
                                    name=f'supervised-{order}',
                                    reuse=reuse)

        return loss

    def _build_supervised_sequential_logprop(self,
                                             source: tf.Tensor,
                                             target: tf.Tensor,
                                             name: str=None,
                                             reuse: bool=False) -> tf.Tensor:
        # logits.shape = (batch * beam, times, vocabulary)
        # labels.shape = (batch * beam, times)
        logits, labels = bytenet_supervised_translator(
            batch_repeat_pack(source),
            batch_repeat_pack(target),
            voca_size=self.dataset.vocabulary_size,
            latent_dim=self.latent_dim,
            num_blocks=self.num_blocks,
            low_memory=True,
            name=name,
            reuse=reuse
        )

        # convert logits to sequence properbilities
        # logits.shape = (batch, samples, times, vocabulary)
        logits = batch_repeat_unpack(logits, repeats=self.beam_size)
        # to convert logits in to logprops over the sequence,
        # first normalize the logits and then select the correct
        # logits dependent on the target. Doing this is actually
        # just a cross entropy transform (except for a sign change
        # build into the cross entropy function).
        logprops = -tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=target
        )
        # reduce logprops over the sequence
        # logprops.shape = (batch, samples)
        return seq_logprop(logprops, mask=target, axis=2)

    def _build_unsupervised_model(self,
                                  source: tf.Tensor,
                                  order: str=None,
                                  reuse: bool=False) -> tf.Tensor:
        name_forward, name_backward, name_scope = (None, None, None)
        if order is not None:
            order_both = f'{order[0]}2{order[2]}'
            name_forward = f'bytenet-{order}'
            name_backward = f'bytenet-{order[::-1]}'
            name_scope = f'unsupervised-bytenet-{order_both}'

        with tf.name_scope(name_scope, 'unsupervised', values=[source]):
            with tf.device('/cpu:0'):
                x = tf.cast(source, tf.int32)

            # compute labels
            with tf.device('/gpu:0'):
                # sample_labels.shape = (batch, beam, times)
                _, sample_labels = bytenet_sampling_translator(
                    x,
                    voca_size=self.dataset.vocabulary_size,
                    latent_dim=self.latent_dim,
                    num_blocks=self.num_blocks,
                    beam_size=self.beam_size,
                    name=name_forward,
                    reuse=reuse,
                    back_prop=False
                )
                sample_labels = tf.stop_gradient(sample_labels)

            with tf.device('/gpu:0'):
                # logprops_xy.shape = (batch, beam)
                logprops_xy = self._build_supervised_sequential_logprop(
                    source=batch_repeat(x, repeats=self.beam_size),
                    target=sample_labels,
                    name=name_forward,
                    reuse=reuse
                )

            with tf.device('/gpu:1'):
                # logprops_xy.shape = (batch, beam)
                logprops_yx = self._build_supervised_sequential_logprop(
                    source=sample_labels,
                    target=batch_repeat(x, repeats=self.beam_size),
                    name=name_backward,
                    reuse=reuse
                )

            with tf.device('/cpu:0'):
                # marginal_logprop.shape (batch, )
                # marginal = sum(P(y'|x) * P(x|y), axis=1)
                #          = sum(exp(logprops_yx) * exp(logprops_xy), axis=1)
                #          = sum(exp(logprops_yx + logprops_xy), axis=1)
                # marginal_logprop = log(marginal)
                # numerial statbility: reduce_logsumexp substracts the max from
                # the input and adds it back in the final output.
                # ln(sum(exp(x - k))) + k = ln(sum(exp(x) * exp(-k))) + k
                #                         = ln(exp(-k) * sum(exp(x))) + k
                #                         = ln(exp(-k)) + ln(sum(exp(x))) + k
                #                         = -k + ln(sum(exp(x))) + k
                #                         = ln(sum(exp(x)))
                marginal_logprop = tf.reduce_logsumexp(
                    logprops_yx + logprops_xy,
                    axis=1
                )

                # cross entropy loss (scalar)
                # the properbility should be maximised, by convention
                # a minimization algorithm is used, this the sign is inverted.
                # batch_loss.shape = (batch, )
                batch_loss = - marginal_logprop

        with tf.device('/cpu:0'):
            loss = cross_entropy_indirect(batch_loss,
                                          name=f'unsupervised-{order_both}')
        return loss

    def loss_model(self,
                   source: tf.Tensor, target: tf.Tensor,
                   reuse: bool=False) -> Tuple[tf.Tensor, LossesType]:
        loss = []

        # place x2y model on /gpu:0
        with tf.device('/gpu:0'):
            loss_x2y = self._build_supervised_model(self.dataset.source,
                                                    self.dataset.target,
                                                    order='x2y',
                                                    reuse=reuse)
            loss.append(loss_x2y)

        # place y2x model on /gpu:1
        with tf.device('/gpu:1'):
            loss_y2x = self._build_supervised_model(self.dataset.target,
                                                    self.dataset.source,
                                                    order='y2x',
                                                    reuse=reuse)
            loss.append(loss_y2x)

        # sum loss and distribute
        loss_sum = tf.add_n(loss)
        return (
            loss_sum,
            {
                'bytenet-x2y': [('/gpu:0', loss_sum)],
                'bytenet-y2x': [('/gpu:1', loss_sum)]
            }
        )

    def train_model(self, reuse: bool=False) -> Tuple[tf.Tensor, LossesType]:
        loss = []

        # compute supervised loss
        supervised_loss, _ = self.loss_model(
            self.dataset.source, self.dataset.target,
            reuse=reuse
        )
        loss.append(supervised_loss)

        # compute unsupervised loss
        if self.dataset_x is not None:
            loss_x2x = self._build_unsupervised_model(self.dataset_x.source,
                                                      order='x2y', reuse=True)
            loss.append(self.dataset_x_loss_factor * loss_x2x)

        if self.dataset_y is not None:
            loss_y2y = self._build_unsupervised_model(self.dataset_y.source,
                                                      order='y2x', reuse=True)
            loss.append(self.dataset_y_loss_factor * loss_y2y)

        # add up losses
        total_loss = tf.add_n(loss)
        tf.summary.scalar('losses/total', total_loss)
        return (
            total_loss,
            {
                'bytenet-x2y': [('/gpu:0', total_loss)],
                'bytenet-y2x': [('/gpu:1', total_loss)]
            }
        )

    def _update_model(self, losses: LossesType, **kwargs) -> List[tf.Tensor]:
        return distributed_tower_optim(losses, **kwargs)

    def inference_model(self,
                        source: tf.Tensor,
                        order='x2y',
                        reuse: bool=False) -> tf.Tensor:
        x = tf.cast(source, tf.int32)

        logits, labels = bytenet_unsupervised_translator(
            x,
            voca_size=self.dataset.vocabulary_size,
            latent_dim=self.latent_dim,
            num_blocks=self.num_blocks,
            name=f'bytenet-{order}',
            reuse=reuse
        )
        return tf.cast(labels, source.dtype)
