
from code.dataset import WMTBilingualNews
from code.model import ByteNet
from code.metric import BleuScore, ModelLoss

# set log level to debug
stf.sg_verbosity(10)

dataset_train = WMTBilingualNews(batch_size=16,
                                 year=2014, source_lang='de', target_lang='en')
dataset_test = WMTBilingualNews(batch_size=16,
                                year=2015, source_lang='de', target_lang='en')

model = ByteNet(dataset_train, save_dir='asset/bytenet_wmt_2014')
model.add_metric(BleuScore(dataset_test))
model.add_metric(ModelLoss(dataset_test))
model.train(max_ep=60, lr=0.0001)
