
import sugartensor as stf

from code.dataset import NLTKComtrans, WMTBilingualNews
from code.model import ByteNet
from code.metric import BleuScore, ModelLoss, OutOfBound

# set log level to debug
stf.sg_verbosity(10)

dataset_train = NLTKComtrans(batch_size=16)
dataset_test = WMTBilingualNews(batch_size=16,
                                year=2014, source_lang='fr', target_lang='en')
model = ByteNet(dataset_train,
                num_blocks=3, latent_dim=400,
                save_dir='asset/bytenet_nltk_comtrans')
model.add_metric(BleuScore(dataset_test))
model.add_metric(ModelLoss(dataset_test))
model.add_metric(OutOfBound(dataset_test))
model.train(max_ep=20, lr=0.0001)
