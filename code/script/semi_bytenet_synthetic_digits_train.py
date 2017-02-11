
import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import SemiSupervisedByteNet

# set log level to debug
stf.sg_verbosity(10)

dataset_x2y = SyntheticDigits(batch_size=16, examples=500, seed=10)
dataset_x = SyntheticDigits(batch_size=16, examples=500, seed=11)

model = SemiSupervisedByteNet(dataset_x2y,
                              dataset_x=dataset_x, dataset_x_loss_factor=0.01,
                              num_blocks=3, latent_dim=20,
                              save_dir='asset/semi_bytenet_synthetic_digits')
model.train(max_ep=300, lr=0.001)
