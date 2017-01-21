
import sugartensor as stf

from dataset import SyntheticDigits
from model import ExportDataset
from model import ByteNet

# set log level to debug
stf.sg_verbosity(10)

dataset = SyntheticDigits(batch_size=16, examples=10, seed=99)
model = ByteNet(dataset, num_blocks=3, latent_dim=20)

export = ExportDataset(dataset)
export.train()

test_predict = model.predict(export.sources)

for i, (source, target, predict) in \
        enumerate(zip(export.sources, export.targets, test_predict)):
    print('  %d  source: %s' % (i, source))
    print('     target: %s' % (target, ))
    print('    predict: %s' % (predict, ))
