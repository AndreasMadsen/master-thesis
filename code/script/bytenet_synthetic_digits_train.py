
import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import ByteNet
from code.metric import MisclassificationRate, ModelLoss

# set log level to debug
stf.sg_verbosity(10)

dataset = SyntheticDigits(batch_size=16, examples=1000, seed=10)
dataset_test = SyntheticDigits(batch_size=50, examples=50, seed=11)

model = ByteNet(dataset, num_blocks=3, latent_dim=20,
                deep_summary=False,
                save_dir='asset/bytenet_synthetic_digits')
model.add_metric(MisclassificationRate(dataset_test))
model.add_metric(ModelLoss(dataset_test))
model.train(max_ep=2000, lr=0.01, log_interval=10, save_interval=30)
