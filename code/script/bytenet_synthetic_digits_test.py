
import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import ExportDataset
from code.model import ByteNet

# set log level to debug
stf.sg_verbosity(10)

dataset_train = SyntheticDigits(batch_size=16, examples=1000, seed=10)
dataset_test = SyntheticDigits(batch_size=10, examples=128, seed=11,
                               shuffle=False, repeat=False)

model = ByteNet(dataset_train,
                num_blocks=3, latent_dim=20,
                deep_summary=False,
                save_dir='hpc_asset/bytenet_synthetic_digits')

translation_tuple = model.predict_from_dataset(dataset_test, samples=10)

for i, (source, target, translation) in zip(range(10), translation_tuple):
    print('  %d  source: %s' % (i, source))
    print('     target: %s' % (target, ))
    print('    predict: %s' % (translation, ))
