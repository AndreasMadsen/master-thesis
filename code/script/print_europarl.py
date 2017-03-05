
from code.dataset import Europarl
from code.model import PrintDataset

dataset = Europarl(min_length=None, max_length=None)
model = PrintDataset(dataset)
model.train()
