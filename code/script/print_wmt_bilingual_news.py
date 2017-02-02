
from code.dataset import WMTBilingualNews
from code.model import PrintDataset

dataset = WMTBilingualNews()
model = PrintDataset(dataset)
model.train()
