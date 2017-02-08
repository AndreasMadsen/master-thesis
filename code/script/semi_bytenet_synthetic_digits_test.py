
import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import ExportDataset
from code.model import SemiSupervisedByteNet

# set log level to debug
stf.sg_verbosity(10)

dataset = SyntheticDigits(examples=10, shuffle=False, seed=99)
model = SemiSupervisedByteNet(dataset,
                              num_blocks=3, latent_dim=20,
                              save_dir='asset/semi_bytenet_synthetic_digits')

export = ExportDataset(dataset)
export.train()

test_predict = model.predict(export.sources)

for i, (source, target, predict) in \
        enumerate(zip(export.sources, export.targets, test_predict)):
    print('  %d  source: %s' % (i, source))
    print('     target: %s' % (target, ))
    print('    predict: %s' % (predict, ))
