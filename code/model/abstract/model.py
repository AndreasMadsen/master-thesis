
import abc
from typing import List

import tensorflow as tf
import sugartensor as stf

from code.dataset.abstract import Dataset
from code.tf_operator import EmbeddingContainer


class Model:
    dataset: Dataset
    _metrics: List['code.metric.abstract.Metric'] = []
    _save_dir: str

    def __init__(self, dataset: Dataset,
                 save_dir: str='asset/unnamed') -> None:
        self.dataset = dataset
        self._save_dir = save_dir
        self.embeddings = EmbeddingContainer()

    def _latest_checkpoint(self) -> str:
        return tf.train.latest_checkpoint(self._save_dir)

    def add_metric(self, metric: 'code.metric.abstract.Metric') -> None:
        self._metrics.append(metric)

    @abc.abstractmethod
    def loss_model(self, x: tf.Tensor, y: tf.Tensor, **kwargs) -> tf.Tensor:
        pass

    def train_model(self) -> tf.Tensor:
        return self.loss_model(self.dataset.source, self.dataset.target)

    def train(self, max_ep: int=20, **kwargs) -> None:
        # build training model
        loss = self.train_model()

        # build eval metrics
        eval_metric = [
            metric.build(self) for metric in self._metrics
        ]

        # save metadata files for embeddings
        self.embeddings.save_metadata(self._save_dir)

        # print tensorboard command
        print(f'tensorboard info:')
        print(f'  using: tensorboard --logdir={self._save_dir}')
        print(f'     on: http://localhost:6006')

        # train
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=sess_config) as sess:
            stf.sg_train(loss=loss,
                         ep_size=self.dataset.num_batch,
                         max_ep=max_ep,
                         eval_metric=eval_metric,
                         early_stop=False,
                         save_dir=self._save_dir,
                         sess=sess,
                         embeds=self.embeddings,
                         **kwargs)

    @abc.abstractmethod
    def inference_model(self, x: tf.Tensor) -> tf.Tensor:
        pass

    def restore(self, session: tf.Session):
        # init session vars
        stf.sg_init(session)

        # restore parameters
        stf.sg_restore(session, self._latest_checkpoint())

    def predict(self, sources: List[str], **kwargs) -> List[str]:
        sources = self.dataset.encode_as_batch(sources)

        # build model
        x = stf.placeholder(dtype=stf.int32, shape=sources.shape)
        label = self.inference_model(x, **kwargs)

        # run graph for translating
        with tf.Session() as sess:
            self.restore(sess)
            pred = sess.run(label, {x: sources})

        return self.dataset.decode_as_batch(pred)
