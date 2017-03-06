
from code.dataset import WMTBilingualNews
from code.model import PrintDataset

dataset = WMTBilingualNews(year=2015,
                           source_lang='de', target_lang='en',
                           min_length=None, max_length=None,
                           shuffle=False, repeat=False)
model = PrintDataset(dataset)
model.train()
