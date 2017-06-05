
import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import ByteNet
from code.metric import MisclassificationRate, ModelLoss

# set log level to debug
stf.sg_verbosity(10)

dataset_train = SyntheticDigits(batch_size=16, examples=128, seed=10)
dataset_test = SyntheticDigits(batch_size=16, examples=128, seed=11)

model = ByteNet(dataset_train,
                version='v1-small',
                num_blocks=3, latent_dim=20,
                deep_summary=False,
                save_dir='asset/bytenet_small_synthetic_digits')

model.add_metric(MisclassificationRate(dataset_train, name='missrate-train'))
model.add_metric(MisclassificationRate(dataset_test, name='missrate-test'))
model.add_metric(ModelLoss(dataset_test, name='model-loss-test'))

model.train(max_ep=300,
            optim='Adam', lr=0.001, beta2=0.999,
            log_interval=10, save_interval=30)
