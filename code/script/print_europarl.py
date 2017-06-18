
from code.dataset import Europarl
from code.model import PrintDataset

dataset = Europarl(batch_size=16,
                   source_lang='de', target_lang='en',
                   min_length=0, max_length=500,
                   external_encoding='build/europarl-max500.tfrecord')

for i, (source, target) in zip(
    range(10),
    sorted(dataset, key=lambda items: min(len(items[0]), len(items[1])))
):
    print('%d  source: %s' % (i, source))
    print('   target: %s' % (target, ))

#model = PrintDataset(dataset)
#model.train()
