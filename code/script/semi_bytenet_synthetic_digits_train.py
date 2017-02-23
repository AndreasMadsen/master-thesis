
import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import SemiSupervisedByteNet
from code.metric import BleuScore, ModelLoss, OutOfBound

# set log level to debug
stf.sg_verbosity(10)

dataset_train = SyntheticDigits(batch_size=8)
dataset_semi = SyntheticDigits(batch_size=8)

model = SemiSupervisedByteNet(dataset_train,
                              dataset_x=dataset_semi, dataset_x_loss_factor=0.01,
                              num_blocks=3, latent_dim=20,
                              save_dir='asset/semi_bytenet_synthetic_digits')
model.train(max_ep=300, lr=0.001)
