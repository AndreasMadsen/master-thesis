
import sugartensor as stf

from code.dataset import NLTKComtrans
from code.model import ExportDataset
from code.model import ByteNet

# set log level to debug
stf.sg_verbosity(10)

dataset = NLTKComtrans(batch_size=16)
model = ByteNet(dataset, num_blocks=3, latent_dim=400)

export = ExportDataset(dataset, limit=10)
export.train()

test_predict = model.predict(export.sources)

for i, (source, target, predict) in \
        enumerate(zip(export.sources, export.targets, test_predict)):
    print('  %d  source: %s' % (i, source))
    print('     target: %s' % (target, ))
    print('    predict: %s' % (predict, ))
