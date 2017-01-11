
from dataset import SyntheticDigits
from model import PrintDataset

dataset = SyntheticDigits(seed=10)
model = PrintDataset(dataset)
model.train()
