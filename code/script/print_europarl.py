
from code.dataset import Europarl
from code.model import PrintDataset

dataset = Europarl(source_lang='de', target_lang='en',
                   min_length=None, max_length=None)
model = PrintDataset(dataset)
model.train()
