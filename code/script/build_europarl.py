
from code.dataset import Europarl

dataset_train = Europarl(batch_size=64,
                         source_lang='de', target_lang='en',
                         min_length=None, max_length=None,
                         external_encoding='build/europarl.tfrecord')

print('build complete :)')
