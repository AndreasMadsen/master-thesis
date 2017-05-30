
import sugartensor as stf

from code.dataset import Europarl, WMTBilingualNews
from code.model import ByteNet
from code.metric import BleuScore, ModelLoss, ObservationLength

# set log level to debug
stf.sg_verbosity(10)

dataset_train = Europarl(batch_size=64,
                         source_lang='de', target_lang='en',
                         min_length=None, max_length=500,
                         external_encoding='build/europarl-max500.tfrecord')

dataset_test = WMTBilingualNews(batch_size=128,
                                year=2015,
                                source_lang='de', target_lang='en',
                                min_length=None, max_length=None,
                                vocabulary=dataset_train.vocabulary,
                                validate=True)

model = ByteNet(dataset_train,
                version='v1-small',
                deep_summary=False,
                save_dir='asset/bytenet_small_europarl_nosummary_max500_adam_debug',
                gpus=4)
model.add_metric(ObservationLength(dataset_train, name='observation-length'))
model.add_metric(BleuScore(dataset_train, name='BLEU-score-train'))
model.add_metric(BleuScore(dataset_test, name='BLEU-score-test'))
model.add_metric(ModelLoss(dataset_test, name='model-loss-test'))
model.train(max_ep=6, optim='Adam', lr=0.0003, beta2=0.999)
