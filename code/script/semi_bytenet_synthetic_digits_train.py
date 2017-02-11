
import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import SemiSupervisedByteNet
from code.metric import MisclassificationRate

# set log level to debug
stf.sg_verbosity(10)

dataset_x2y = SyntheticDigits(batch_size=16, examples=500, seed=10)
dataset_x = SyntheticDigits(batch_size=16, examples=500, seed=11)

dataset_test_x2y = SyntheticDigits(batch_size=50, examples=50, seed=12)
missrate = MisclassificationRate(test_dataset_x,
                                 name="misclassification-rate-x2y")

model = SemiSupervisedByteNet(dataset_x2y,
                              dataset_x=dataset_x, dataset_x_loss_factor=0.01,
                              num_blocks=3, latent_dim=20,
                              save_dir='asset/semi_bytenet_synthetic_digits')
model.add_metric(missrate)
model.train(max_ep=300, lr=0.001, log_interval=30)
