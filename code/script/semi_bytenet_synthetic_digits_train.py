
import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import SemiSupervisedByteNet

# set log level to debug
stf.sg_verbosity(10)

dataset_train = SyntheticDigits(batch_size=16)
dataset_semi = SyntheticDigits(batch_size=16)

model = SemiSupervisedByteNet(dataset_train,
                              dataset_x=dataset_semi,
                              dataset_x_loss_factor=0.1,
                              num_blocks=3, latent_dim=20,
                              gpus=4,
                              save_dir='asset/semi_bytenet_synthetic_digits_gpu4')
model.train(max_ep=300, lr=0.001)
