
from typing import Tuple
import abc

import tensorflow as tf
import numpy as np


class SequenceQueue:
    need_data: bool  # public
    dtype: np.unsignedinteger
    batch_size: int
    observations: int
    name: str
    shuffle: bool
    seed: int
    repeat: bool

    def __init__(self,
                 need_data: bool,
                 observations: int,
                 dtype: np.unsignedinteger,
                 batch_size: int=32,
                 name: str='unamed',
                 shuffle: bool=True, seed: int=None,
                 repeat: bool=True):
        self.need_data = need_data
        self.observations = observations
        self.dtype = dtype
        self.batch_size = batch_size
        self.name = name
        self.shuffle = shuffle
        self.seed = seed
        self.repeat = repeat

    @abc.abstractmethod
    def write(self,
              length: int, source: np.ndarray, target: np.ndarray) -> None:
        pass

    @abc.abstractmethod
    def read(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        pass
