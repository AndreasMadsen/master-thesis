
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
    distributed_tower_optim, \
    tower_scope, mean_n, \
    cross_entropy_summary


class SemiSupervisedByteNet(Model):
    _gpus: int
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
                 gpus=2,
                 **kwargs) -> None:
        super().__init__(dataset_x2y, save_dir=save_dir, **kwargs)

        if gpus < 2 or gpus % 2 != 0:
            raise ValueError(f'gpus ({gpus}) must be even an >= 2')
        self._gpus = gpus

        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.beam_size = beam_size

        self.dataset_x = dataset_x
        self.dataset_x_loss_factor = dataset_x_loss_factor
        self.dataset_y = dataset_y
        self.dataset_y_loss_factor = dataset_y_loss_factor

    def _build_supervised_model(self,
                                source_all: tf.Tensor, target_all: tf.Tensor,
                                order: str=None,
                                gpus: List[int]=[0],
                                reuse: bool=False) -> tf.Tensor:
        name, name_scope = (None, None)
        if order is not None:
            name = f'bytenet-{order}'
            name_scope = f'supervised-bytenet-{order}'

        with tf.name_scope(name_scope, 'supervised',
                           values=[source_all, target_all]):
            # putting the split and join on the cpu is extreamly important for
            # minimizing the syncronization time.
            with tf.device('/cpu:0'):
                source_split = tf.split(source_all, len(gpus), axis=0)
                target_split = tf.split(target_all, len(gpus), axis=0)

            losses = []

            for (index, device), source, target in zip(
                    tower_scope(gpus, reuse=reuse),
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
                    name=name,
                    reuse=reuse
                )

                losses.append((
                    device,
                    cross_entropy_direct(logits, y, name=f'supervised-{order}')
                ))

            # join the losses
            with tf.device('/cpu:0'):
                total_loss = mean_n([loss for _, loss in losses])

        with tf.device('/cpu:0'):
            total_loss = cross_entropy_summary(total_loss,
                                               name=f'supervised-{order}',
                                               reuse=reuse)

        return (total_loss, losses)

    def _build_unsupervised_sampler(self,
                                    source_split: List[tf.Tensor],
                                    name: str=None,
                                    gpus: List[int]=[0],
                                    reuse: bool=False
                                    ) -> List[tf.Tensor]:
        sample_labels_split = []
        for (index, device), source in zip(
                tower_scope(gpus, reuse=reuse),
                source_split
        ):
            # sample_labels.shape = (batch, beam, times)
            _, sample_labels = bytenet_sampling_translator(
                source,
                voca_size=self.dataset.vocabulary_size,
                latent_dim=self.latent_dim,
                num_blocks=self.num_blocks,
                beam_size=self.beam_size,
                name=name,
                reuse=reuse,
                back_prop=False
            )
            sample_labels = tf.stop_gradient(sample_labels)
            sample_labels_split.append(sample_labels)

        return sample_labels_split

    def _build_supervised_sequential_logprop(self,
                                             source_split: List[tf.Tensor],
                                             target_split: List[tf.Tensor],
                                             name: str=None,
                                             gpus: List[int]=[0],
                                             reuse: bool=False
                                             ) -> List[tf.Tensor]:
        seq_logprop_split = []

        for (index, device), source, target in zip(
                tower_scope(gpus, reuse=reuse),
                source_split, target_split
        ):
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
            seq_logprop_split.append(
                seq_logprop(logprops, mask=target, axis=2)
            )

        return seq_logprop_split

    def _build_unsupervised_model(self,
                                  source: tf.Tensor,
                                  forward_gpus=[0],
                                  forward_order='x2y',
                                  backward_gpus=[1],
                                  backward_order='y2x',
                                  reuse: bool=False) -> tf.Tensor:
        order_both = f'{forward_order[0]}2{backward_order[-1]}'
        name_forward = f'bytenet-{forward_order}'
        name_backward = f'bytenet-{backward_order}'
        name_scope = f'unsupervised-bytenet-{order_both}'

        with tf.name_scope(name_scope, 'unsupervised', values=[source]):
            # putting the split and join on the cpu is extreamly important for
            # minimizing the syncronization time.
            with tf.device('/cpu:0'):
                x_cast = tf.cast(source, tf.int32)
                x_split = tf.split(x_cast, self._gpus // 2, axis=0)
                x_repeat_split = []
                for x in x_split:
                    x_repeat = batch_repeat(x, repeats=self.beam_size)
                    x_repeat_split.append(x_repeat)

            # sample_labels.shape = (batch, beam, times)
            sample_labels_split = self._build_unsupervised_sampler(
                source_split=x_split,
                name=name_forward,
                gpus=forward_gpus,
                reuse=reuse
            )

            # logprops_xy.shape = (batch, beam)
            logprops_xy_split = self._build_supervised_sequential_logprop(
                source_split=x_repeat_split,
                target_split=sample_labels_split,
                name=name_forward,
                gpus=forward_gpus,
                reuse=reuse
            )

            # logprops_yx.shape = (batch, beam)
            logprops_yx_split = self._build_supervised_sequential_logprop(
                source_split=sample_labels_split,
                target_split=x_repeat_split,
                name=name_backward,
                gpus=backward_gpus,
                reuse=reuse
            )

            # since logprops_yx and logprops_xy are on diffrent GPUs
            # there is no sutiable device to compute the margial logpro on.
            # however because the computation is fairly minimal it can just
            # be on the cpu.
            with tf.device('/cpu:0'):
                loss_split = []
                for logprops_xy, logprops_yx in zip(
                    logprops_xy_split, logprops_yx_split
                ):
                    # marginal_logprop.shape (batch, )
                    # marginal = sum(P(y'|x) * P(x|y))
                    #          = sum(exp(logprops_yx) * exp(logprops_xy))
                    #          = sum(exp(logprops_yx + logprops_xy))
                    # marginal_logprop = log(marginal)
                    # numerial statbility: reduce_logsumexp substracts the max
                    # from the input and adds it back in the final output.
                    # ln(sum(exp(x - k))) + k = ln(sum(exp(x) * exp(-k))) + k
                    #                       = ln(exp(-k) * sum(exp(x))) + k
                    #                       = ln(exp(-k)) + ln(sum(exp(x))) + k
                    #                       = -k + ln(sum(exp(x))) + k
                    #                       = ln(sum(exp(x)))
                    marginal_logprop = tf.reduce_logsumexp(
                        logprops_yx + logprops_xy,
                        axis=1
                    )

                    # calculate cross entropy from logprops
                    # (change sign, sum, numerical check)
                    loss = cross_entropy_indirect(
                        marginal_logprop, name=f'unsupervised-{order_both}'
                    )
                    loss_split.append(loss)

                # sum up all losses
                loss_total = mean_n(loss_split)

        with tf.device('/cpu:0'):
            loss_total = cross_entropy_summary(
                loss_total, name=f'unsupervised-{order_both}'
            )

        return loss_total, loss_split

    def loss_model(self,
                   source: tf.Tensor, target: tf.Tensor, length: tf.Tensor,
                   reuse: bool=False) -> Tuple[tf.Tensor, LossesType]:
        # place x2y model on even GPUs
        loss_x2y, losses_x2y = self._build_supervised_model(
            self.dataset.source, self.dataset.target,
            order='x2y',
            gpus=range(0, self._gpus, 2),
            reuse=reuse
        )

        # place y2x model on odd GPUs
        loss_y2x, losses_y2x = self._build_supervised_model(
            self.dataset.target, self.dataset.source,
            order='y2x',
            gpus=range(1, self._gpus, 2),
            reuse=reuse
        )

        # add together the losses for each split, not strictly necessary
        # because the models are completely seperate, but logically it
        # makes more sense.
        losses = [
            x2y_loss + y2x_loss
            for (_, x2y_loss), (_, y2x_loss) in zip(losses_x2y, losses_y2x)
        ]

        # sum loss and distribute
        return (
            loss_x2y + loss_y2x,
            {
                'bytenet-x2y': [
                    (device, loss)
                    for (device, _), loss in zip(losses_x2y, losses)
                ],
                'bytenet-y2x': [
                    (device, loss)
                    for (device, _), loss in zip(losses_y2x, losses)
                ]
            }
        )

    def train_model(self, reuse: bool=False) -> Tuple[tf.Tensor, LossesType]:
        #
        # compute supervised loss
        #
        supervised_loss, supervised_losses = self.loss_model(
            self.dataset.source, self.dataset.target, self.dataset.length,
            reuse=reuse
        )

        #
        # compute unsupervised loss
        #
        total_loss_x2x, loss_x2x_split = (
            tf.constant(0, dtype=tf.float32),
            [tf.constant(0, dtype=tf.float32)] * (self._gpus // 2)
        )
        if self.dataset_x is not None:
            # compute unsupervised loss
            total_loss_x2x, loss_x2x_split = self._build_unsupervised_model(
                self.dataset_x.source,
                forward_order='x2y',
                forward_gpus=range(0, self._gpus, 2),
                backward_order='y2x',
                backward_gpus=range(1, self._gpus, 2),
                reuse=True
            )
            # multiply by unsupervised factor
            total_loss_x2x = self.dataset_x_loss_factor * total_loss_x2x
            loss_x2x_split = [
                self.dataset_x_loss_factor * loss_x2x
                for loss_x2x in loss_x2x_split
            ]

        total_loss_y2y, loss_y2y_split = (
            tf.constant(0, dtype=tf.float32),
            [tf.constant(0, dtype=tf.float32)] * (self._gpus // 2)
        )
        if self.dataset_y is not None:
            # compute unsupervised loss
            total_loss_y2y, loss_y2y_split = self._build_unsupervised_model(
                self.dataset_y.source,
                forward_order='y2x',
                forward_gpus=range(1, self._gpus, 2),
                backward_order='x2y',
                backward_gpus=range(0, self._gpus, 2),
                reuse=True
            )
            # multiply by unsupervised factor
            total_loss_y2y = self.dataset_y_loss_factor * total_loss_y2y
            loss_y2y_split = [
                self.dataset_y_loss_factor * loss_y2y
                for loss_y2y in loss_y2y_split
            ]

        #
        # add up losses
        #
        total_loss = tf.add_n([
            supervised_loss,
            total_loss_x2x,
            total_loss_y2y
        ])
        if not reuse:
            tf.summary.scalar('losses/total', total_loss)

        # distribute update:
        #   let the supervised loss dictate how the update is distrubuted
        #   since that contains all the information about how x2y and y2x
        #   are distributed on the GPUs.
        # loss[i] = supervised_loss[i] + x2x_loss[i] + y2y_loss[i]
        # {
        #   'bytenet-x2y': [('/gpu:0', loss[0]), ('/gpu:2', loss[1])], -- even
        #   'bytenet-y2x': [('/gpu:1', loss[0]), ('/gpu:3', loss[1])]  -- odd
        # }
        return (
            total_loss,
            {
                'bytenet-x2y': [
                    (device, tf.add_n([supervised_loss, x2x_loss, y2y_loss]))
                    for (device, supervised_loss), x2x_loss, y2y_loss
                    in zip(
                        supervised_losses['bytenet-x2y'],
                        loss_x2x_split, loss_y2y_split
                    )
                ],
                'bytenet-y2x': [
                    (device, tf.add_n([supervised_loss, x2x_loss, y2y_loss]))
                    for (device, supervised_loss), x2x_loss, y2y_loss
                    in zip(
                        supervised_losses['bytenet-y2x'],
                        loss_x2x_split, loss_y2y_split
                    )
                ]
            }
        )

    def _update_model(self, losses: LossesType, **kwargs) -> List[tf.Tensor]:
        return distributed_tower_optim(losses, **kwargs)

    def inference_model(self,
                        source: tf.Tensor, length: tf.Tensor,
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
