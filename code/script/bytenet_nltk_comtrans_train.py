
import sugartensor as stf

from code.dataset import NLTKComtrans
from code.model import ByteNet

# set log level to debug
stf.sg_verbosity(10)

dataset = NLTKComtrans(batch_size=16)
model = ByteNet(dataset, num_blocks=3, latent_dim=400)
model.train(lr=0.0001)
