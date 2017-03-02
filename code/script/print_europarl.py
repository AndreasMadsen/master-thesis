
from code.dataset import Europarl
from code.model import PrintDataset

dataset = Europarl()
model = PrintDataset(dataset)
model.train()
