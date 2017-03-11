
import sugartensor as stf

from code.dataset import SyntheticDigits
from code.model import ExportDataset
from code.model import ByteNet

# set log level to debug
stf.sg_verbosity(10)

dataset = SyntheticDigits(examples=10, seed=99, shuffle=False, repeat=False)
model = ByteNet(dataset, num_blocks=3, latent_dim=20,
                save_dir='asset/bytenet_synthetic_digits')

predict_sample = model.sample_from_dataset(dataset, samples=5)

for i, (source, target, predicts) in enumerate(predict_sample):
    print('  %d  source: %s' % (i, source))
    print('     target: %s' % (target, ))
    print('    predict:')
    for predict_i, predict in enumerate(predicts):
        print('        | %d: %s' % (predict_i, predict))
