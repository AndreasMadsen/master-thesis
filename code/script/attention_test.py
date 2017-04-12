
from code.dataset import SyntheticDigits
from code.model import Attention

import tensorflow as tf
import sugartensor as stf

# set log level to debug
stf.sg_verbosity(10)

dataset_train = SyntheticDigits(batch_size=16, examples=1000, seed=10)
dataset_test = SyntheticDigits(batch_size=16, examples=10, seed=11,
                               shuffle=False, repeat=False)

model = Attention(dataset_train)

translation_tuple = model.predict_from_dataset(dataset_test)

for i, (source, target, translation) in zip(range(128), translation_tuple):
    print('  %d  source: %s' % (i, source))
    print('     target: %s' % (target, ))
    print('    predict: %s' % (translation, ))
