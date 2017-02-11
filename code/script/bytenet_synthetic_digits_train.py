
import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import ByteNet
from code.metric import MisclassificationRate

# set log level to debug
stf.sg_verbosity(10)

dataset = SyntheticDigits(batch_size=16, examples=1000, seed=10)
dataset_test = SyntheticDigits(batch_size=50, examples=50, seed=11)

missrate = MisclassificationRate(dataset_test,
                                 name="misclassification-rate")


model = ByteNet(dataset, num_blocks=3, latent_dim=20,
                save_dir='asset/bytenet_synthetic_digits')
model.add_metric(missrate)
model.train(max_ep=2000, lr=0.01, log_interval=15)
