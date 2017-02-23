
import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import SemiSupervisedByteNet
from code.metric import BleuScore, ModelLoss, OutOfBound

# set log level to debug
stf.sg_verbosity(10)

dataset_train = SyntheticDigits(batch_size=8)
dataset_semi = SyntheticDigits(batch_size=8)
dataset_test = SyntheticDigits(batch_size=16)

model = SemiSupervisedByteNet(dataset_train,
                              dataset_x=dataset_semi, dataset_x_loss_factor=0.01,
                              num_blocks=3, latent_dim=20,
                              save_dir='asset/semi_bytenet_synthetic_digits')

translation_tuple = model.predict_from_dataset(dataset_test)

for i, (source, target, translation) in zip(range(16), translation_tuple):
    print('  %d  source: %s' % (i, source))
    print('     target: %s' % (target, ))
    print('    predict: %s' % (translation, ))
