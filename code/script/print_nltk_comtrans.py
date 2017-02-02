
from code.dataset import NLTKComtrans
from code.model import PrintDataset

dataset = NLTKComtrans()
model = PrintDataset(dataset)
model.train()
