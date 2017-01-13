
import sugartensor as stf

from dataset import SyntheticDigits
from model import ByteNet

# set log level to debug
stf.sg_verbosity(10)

dataset = SyntheticDigits(batch_size=16, examples=100, seed=10)
model = ByteNet(dataset, num_blocks=3, latent_dim=20)
model.train(max_ep=100)
