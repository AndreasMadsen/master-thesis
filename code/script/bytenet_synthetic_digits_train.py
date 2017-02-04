
import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import ByteNet

# set log level to debug
stf.sg_verbosity(10)

dataset = SyntheticDigits(batch_size=16, examples=1000, seed=10)
model = ByteNet(dataset, num_blocks=3, latent_dim=20)
model.train(max_ep=100, lr=0.0001)
