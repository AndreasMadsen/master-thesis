
from typing import TypeVar, Any, List, Tuple, Iterator
import abc
import os
import os.path as path

import numpy as np
import tensorflow as tf
import sugartensor as stf

from code.dataset.abstract import Dataset
from code.tf_operator import \
    EmbeddingContainer, \
    tower_optim, \
    basic_train, \
    flatten_losses

LossesType = Iterator[Any]


class Model:
    dataset: Dataset
    _metrics: List['code.metric.abstract.Metric'] = []
    _save_dir: str
    _deep_summary: bool

    def __init__(self, dataset: Dataset,
                 deep_summary=True,
                 save_dir: str='asset/unnamed') -> None:
        self.dataset = dataset
        self._deep_summary = deep_summary
        self.set_save_dir(save_dir)

        self.embeddings = EmbeddingContainer()

    def set_save_dir(self, save_dir):
        if 'BASE_SAVE_DIR' in os.environ:
            self._save_dir = path.join(os.environ['BASE_SAVE_DIR'], save_dir)
        else:
            self._save_dir = save_dir

    def get_save_dir(self):
        return self._save_dir

    def _latest_checkpoint(self) -> str:
        return tf.train.latest_checkpoint(self._save_dir)

    def _options_context(self) -> Any:
        return stf.sg_context(summary=self._deep_summary)

    def add_metric(self, metric: 'code.metric.abstract.Metric') -> None:
        self._metrics.append(metric)

    @abc.abstractmethod
    def loss_model(self, x: tf.Tensor, y: tf.Tensor, length: tf.Tensor,
                   **kwargs) -> LossesType:
        pass

    def train_model(self, **kwargs) -> LossesType:
        return self.loss_model(self.dataset.source, self.dataset.target,
                               self.dataset.length,
                               **kwargs)

    def train(self, max_ep: int=20,
              reuse: bool=False,
              allow_soft_placement: bool=True,
              log_device_placement: bool=False,
              log_interval=60, save_interval=600,
              profile=0,
              tqdm=True,
              lr=0.001,
              **kwargs) -> None:
        # build training model
        with self._options_context():
            loss, losses = self.train_model(reuse=reuse)
        losses_ops = flatten_losses(losses)

        # build eval metrics
        eval_metric = [
            metric.build(self) for metric in self._metrics
        ]

        # compute update
        with tf.variable_scope('train', reuse=reuse,
                               values=[loss] + losses_ops + eval_metric):
            with self._options_context():
                update = self._update_model(losses, lr=lr, **kwargs)

        # save metadata files for embeddings
        self.embeddings.save_metadata(self._save_dir)

        # print tensorboard command
        print(f'tensorboard info:')
        print(f'  using: tensorboard --logdir={self._save_dir}')
        print(f'     on: http://localhost:6006')

        # train
        sess_config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,
                                     log_device_placement=log_device_placement)
        # XLA will be used on DTU-HPC and will JIT compile GPU kernels
        if 'TF_USE_XLA' in os.environ:
            sess_config.graph_options.optimizer_options.global_jit_level = \
                tf.OptimizerOptions.ON_1

        self._train_loop(loss, update,
                         ep_size=self.dataset.num_batch,
                         max_ep=max_ep,
                         eval_metric=eval_metric,
                         early_stop=False,
                         save_dir=self._save_dir,
                         config=sess_config,
                         tqdm=tqdm,
                         profile=profile,
                         log_interval=log_interval,
                         save_interval=save_interval,
                         lr=lr)

    def _update_model(self, losses: LossesType, **kwargs) -> List[tf.Tensor]:
        return tower_optim(losses, **kwargs)

    def _train_loop(self,
                    loss: tf.Tensor, update_op: List[tf.Tensor],
                    **kwargs) -> None:
        basic_train(loss, update_op, **kwargs)

    def restore(self, session: tf.Session):
        # init session vars
        stf.sg_init(session)

        # restore parameters
        stf.sg_restore(session, self._latest_checkpoint())

    @abc.abstractmethod
    def greedy_inference_model(self, x: tf.Tensor,
                               length: tf.Tensor) -> tf.Tensor:
        pass

    @abc.abstractmethod
    def sample_inference_model(self, x: tf.Tensor, length: tf.Tensor,
                               samples: int=1) -> tf.Tensor:
        pass

    def inference_model(self, x: tf.Tensor, length: tf.Tensor,
                        samples: int=1, **kwargs) -> tf.Tensor:
        if samples == 1:
            return self.greedy_inference_model(x, length, **kwargs)
        else:
            # labels.shape = (batch, samples, time)
            labels = self.sample_inference_model(x, length,
                                                 samples=samples, **kwargs)
            return labels[:, 0]

    def _predict_from_dataset_queue(self, dataset, **kwargs):
        # build inference_model
        with self._options_context():
            label = self.inference_model(dataset.source, dataset.length,
                                         **kwargs)

        with tf.Session() as sess:
            self.restore(sess)
            with stf.sg_queue_context():
                for batch in range(dataset.num_batch):
                    source, target, translation = sess.run(
                        (dataset.source, dataset.target, label)
                    )

                    yield source, target, translation

    def _predict_from_dataset_feed(self, dataset, **kwargs):
        observations = dataset.num_observation

        # build inference_model
        x = stf.placeholder(dtype=tf.int32, shape=(None, None))
        length = stf.placeholder(dtype=tf.int32, shape=(None, ))
        with self._options_context():
            label = self.inference_model(x, length, **kwargs)

        with tf.Session() as sess:
            self.restore(sess)
            with stf.sg_queue_context():
                for batch in dataset.batch_iterator():
                    # seperate source and target and encode batch
                    source, target = zip(*batch)
                    source, source_len = self.dataset.encode_as_batch(source)
                    target, target_len = self.dataset.encode_as_batch(target)

                    # translate batch
                    translation = sess.run(label, {
                        x: source,
                        length: np.maximum(source_len, target_len)
                    })

                    # return results
                    yield source, target, translation

    def predict_from_dataset(self, dataset: Dataset,
                             show_eos: bool=True,
                             use_queue: bool=True,
                             **kwargs) -> Iterator[Tuple[str, str, str]]:
        if use_queue:
            evaluator = self._predict_from_dataset_queue(dataset, **kwargs)
        else:
            evaluator = self._predict_from_dataset_feed(dataset, **kwargs)

        for source, target, translation in evaluator:

            # unpack and decode result
            yield from zip(
                self.dataset.decode_as_batch(source,
                                             show_eos=show_eos),
                self.dataset.decode_as_batch(target,
                                             show_eos=show_eos),
                self.dataset.decode_as_batch(translation,
                                             show_eos=show_eos)
            )

    def predict_from_str(self, sources: List[str],
                         show_eos: bool=True,
                         batch_size: int=16,
                         **kwargs) -> Iterator[str]:
        source, source_len = self.dataset.encode_as_batch(sources)
        observations = source.shape[0]

        # build model
        x = stf.placeholder(dtype=tf.int32, shape=(None, *source.shape[1:]))
        length = stf.placeholder(dtype=tf.int32, shape=(None, ))
        with self._options_context():
            label = self.inference_model(x, length, **kwargs)

        # run graph for translation
        with tf.Session() as sess:
            self.restore(sess)
            for start, end in zip(
                range(0, observations, batch_size),
                range(batch_size, observations + batch_size, batch_size)
            ):
                pred = sess.run(label, {
                    x: source[start:end],
                    length: source_len[start:end]
                })

                yield from self.dataset.decode_as_batch(pred,
                                                        show_eos=show_eos)

    def sample_from_dataset(self, dataset: Dataset,
                            show_eos: bool=True, samples: int=1,
                            **kwargs) -> Iterator[Tuple[str, str, str]]:

        # build inference_model
        with self._options_context():
            label = self.sample_inference_model(dataset.source,
                                                samples=samples,
                                                **kwargs)

        # run graph for translation
        with tf.Session() as sess:
            self.restore(sess)
            with stf.sg_queue_context():
                for batch in range(dataset.num_batch):
                    source, target, translation_samples = sess.run(
                        (dataset.source, dataset.target, label)
                    )

                    yield from zip(
                        self.dataset.decode_as_batch(source,
                                                     show_eos=show_eos),
                        self.dataset.decode_as_batch(target,
                                                     show_eos=show_eos),
                        (self.dataset.decode_as_batch(batch,
                                                      show_eos=show_eos)
                         for batch in translation_samples)
                    )

    def sample_from_str(self, sources: List[str],
                        show_eos: bool=True, samples: int=1,
                        **kwargs) -> Iterator[List[str]]:
        source, source_len = self.dataset.encode_as_batch(sources)
        observations = source.shape[0]

        # build model
        x = stf.placeholder(dtype=tf.int32, shape=(None, *source.shape[1:]))
        length = stf.placeholder(dtype=tf.int32, shape=(None, ))
        with self._options_context():
            label = self.sample_inference_model(x, length, **kwargs)

        # run graph for translation
        with tf.Session() as sess:
            self.restore(sess)
            for start, end in zip(
                range(0, observations, batch_size),
                range(batch_size, observations + batch_size, batch_size)
            ):
                pred = sess.run(label, {
                    x: source[start:end],
                    length: source_len[start:end]
                })

                yield from (
                    self.dataset.decode_as_batch(batch, show_eos=show_eos)
                    for batch in pred
                )
