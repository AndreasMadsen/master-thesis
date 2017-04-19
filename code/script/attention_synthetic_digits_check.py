
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

model = Attention(
    dataset_train,
    num_blocks=3, latent_dim=20,
    gpus=1,
    deep_summary=False,
    save_dir=f'asset/attention_synthetic_digits_grid/' +
             f'train_{args.train_size}'
)

translation_tuple = model.predict_from_dataset(dataset_train)

for i, (source, target, translation) in zip(range(128), translation_tuple):
    print('  %d  source: %s' % (i, source))
    print('     target: %s' % (target, ))
    print('    predict: %s' % (translation, ))
