
from code.dataset import SyntheticDigits
from code.model import Attention

import tensorflow as tf
import sugartensor as stf

# set log level to debug
stf.sg_verbosity(10)

dataset_train = SyntheticDigits(batch_size=16, examples=1000, seed=10)

model = Attention(dataset_train, deep_summary=False)
model.train(max_ep=200, optim='Adam', lr=1e-4,
            log_interval=10, save_interval=30)
