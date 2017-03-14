
from code.dataset import Europarl
from code.model import PrintDataset

dataset = Europarl(batch_size=16,
                   source_lang='de', target_lang='en',
                   min_length=None, max_length=None,
                   external_encoding='build/europarl-full.tfrecord')
model = PrintDataset(dataset)
model.train()
