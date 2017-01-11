
from dataset import NLTKComtrans
from model import PrintDataset

dataset = NLTKComtrans()
model = PrintDataset(dataset)
model.train()
