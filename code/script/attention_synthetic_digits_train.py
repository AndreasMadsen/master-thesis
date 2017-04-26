
import argparse

import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import Attention
from code.metric import MisclassificationRate, ModelLoss

# set log level to debug
stf.sg_verbosity(10)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--train-size", help="number of train observations",
                    default=128, type=int)
args = parser.parse_args()

# argument grid
# train_size = [64, 128, 512]

print('Parameters')
print(f'  train-size: {args.train_size}')

# train model
dataset_train = SyntheticDigits(batch_size=16, examples=args.train_size,
                                seed=10)

dataset_test = SyntheticDigits(batch_size=128, examples=1024,
                               seed=12)

model = Attention(
    dataset_train,
    num_blocks=3, latent_dim=20,
    gpus=1,
    deep_summary=False,
    save_dir=f'asset/attention_synthetic_digits_baseline/' +
             f'train_{args.train_size}'
)
model.add_metric(MisclassificationRate(dataset_test))
model.add_metric(ModelLoss(dataset_test))
model.train(max_ep=1000, optim='Adam', lr=1e-4,
            log_interval=20, save_interval=60)
