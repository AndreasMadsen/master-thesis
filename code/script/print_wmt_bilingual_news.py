
from dataset import WMTBilingualNews
from model import PrintDataset

dataset = WMTBilingualNews()
model = PrintDataset(dataset)
model.train()
