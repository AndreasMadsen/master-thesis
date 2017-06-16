
from code.dataset import Europarl

dataset_train = Europarl(batch_size=64,
                         source_lang='de', target_lang='en',
                         min_length=25, max_length=500,
                         external_encoding='build/europarl-min25-max500.tfrecord')

print('build complete :)')
