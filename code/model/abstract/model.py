
import abc

import tensorflow as tf
import sugartensor as stf

from code.dataset import Dataset


class Model:
    dataset: Dataset
    _save_dir: str

    def __init__(self, dataset: Dataset,
                 save_dir: str='asset/unnamed') -> None:
        self.dataset = dataset
        self._save_dir = save_dir

    @abc.abstractmethod
    def _model_loss(self) -> tf.Tensor:
        pass

    def _latest_checkpoint(self) -> str:
        return tf.train.latest_checkpoint(self._save_dir)

    def train(self, max_ep: int=20, **kwargs) -> None:
        loss = self._model_loss()

        # print tensorboard command
        print(f'tensorboard info:')
        print(f'  using: tensorboard --logdir={self._save_dir}')
        print(f'     on: http://localhost:6006')

        # train
        sess_config = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=sess_config) as sess:
            stf.sg_train(log_interval=30,
                         loss=loss,
                         ep_size=self.dataset.num_batch,
                         max_ep=max_ep,
                         early_stop=False,
                         save_dir=self._save_dir,
                         sess=sess,
                         **kwargs)

    @abc.abstractmethod
    def predict(self) -> None:
        pass
