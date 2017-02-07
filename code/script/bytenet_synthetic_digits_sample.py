
import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import ExportDataset
from code.model import ByteNet

# set log level to debug
stf.sg_verbosity(10)

dataset = SyntheticDigits(examples=10, shuffle=False, seed=99)
model = ByteNet(dataset, num_blocks=3, latent_dim=20,
                save_dir='asset/bytenet_synthetic_digits')

export = ExportDataset(dataset)
export.train()

samples_predict = model.sample(export.sources, samples=5)

for i, (source, target, predicts) in \
        enumerate(zip(export.sources, export.targets, samples_predict)):
    print('  %d  source: %s' % (i, source))
    print('     target: %s' % (target, ))
    print('    predict:')
    for predict_i, predict in enumerate(predicts):
        print('        | %d: %s' % (predict_i, predict))
