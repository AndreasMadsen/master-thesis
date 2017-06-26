
import argparse

import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import SemiSupervisedByteNet
from code.metric import MisclassificationRate, ModelLoss

# set log level to debug
stf.sg_verbosity(10)

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--iteration", help="the statistical trial number",
                    default=1, type=int)
parser.add_argument("--train-size", help="number of train observations",
                    default=128, type=int)
parser.add_argument("--semi-size", help="number of unlabeled observations",
                    default=128, type=int)
parser.add_argument("--semi-factor", help="importance factor of semi loss",
                    default=0.1, type=float)
args = parser.parse_args()

# argument grid
# train_size = [64, 128, 256, 512, 1024]
# semi_size = [512, 1024]
# semi_factor = [0.2, 0.1, 0.01]

print('Parameters')
print(f'  iteration: {args.iteration}')
print(f'  train-size: {args.train_size}')
print(f'  semi-size: {args.semi_size}')
print(f'  semi-factor: {args.semi_factor}')

# train model
dataset_train = SyntheticDigits(batch_size=8, examples=args.train_size,
                                seed=10 * args.iteration)
if args.semi_size > 0:
    dataset_semi = SyntheticDigits(batch_size=8, examples=args.semi_size,
                                   seed=10 * args.iteration + 1)
else:
    dataset_semi = None

dataset_test = SyntheticDigits(batch_size=128, examples=1024,
                               seed=10 * args.iteration + 2)

model = SemiSupervisedByteNet(
    dataset_train,
    dataset_x=dataset_semi, dataset_x_loss_factor=args.semi_factor,
    num_blocks=3, latent_dim=20, beam_size=5,
    gpus=2,
    deep_summary=False,
    save_dir=f'asset/semi_bytenet_synthetic_digits_grid_stats/' +
             f'train_{args.train_size}_semi_{args.semi_size}_' +
             f'factor_{args.semi_factor}_iter{args.iteration}'
)
model.add_metric(MisclassificationRate(dataset_test))
model.add_metric(ModelLoss(dataset_test))
model.train(max_ep=300,
            optim='Adam', lr=0.001, beta2=0.999)
