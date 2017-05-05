
import sugartensor as stf

from code.dataset import WMTBilingualNews
from code.model import ByteNet
from code.metric import BleuScore, ModelLoss

# set log level to debug
stf.sg_verbosity(10)

dataset_train = WMTBilingualNews(batch_size=16,
                                 year=2014,
                                 source_lang='de', target_lang='en',
                                 min_length=None, max_length=None)

dataset_test = WMTBilingualNews(batch_size=128,
                                year=2015,
                                source_lang='de', target_lang='en',
                                min_length=None, max_length=None,
                                vocabulary=dataset_train.vocabulary,
                                validate=True)

model = ByteNet(dataset_train, save_dir='asset/bytenet_wmt_2014_xla_compare/bytenet_wmt_2014_gpu1_noxla_profile',
                deep_summary=False, gpus=1)
model.add_metric(BleuScore(dataset_train, name='BLEU-score-train'))
model.add_metric(BleuScore(dataset_test, name='BLEU-score-test'))
model.add_metric(ModelLoss(dataset_test, name='model-loss-test'))
model.train(max_ep=10, profile=100, lr=0.0001)
