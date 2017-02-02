
from code.dataset import SyntheticDigits
from code.model import PrintDataset

dataset = SyntheticDigits(seed=10)
model = PrintDataset(dataset)
model.train()
