
import sugartensor as stf

from code.dataset import NLTKComtrans, WMTBilingualNews
from code.model import SemiSupervisedByteNet

# set log level to debug
stf.sg_verbosity(10)

dataset_train = NLTKComtrans(batch_size=8)
dataset_semi = WMTBilingualNews(batch_size=8,
                                year=2013, source_lang='fr', target_lang='en',
                                vocabulary=dataset_train.vocabulary,
                                validate=True)

model = SemiSupervisedByteNet(dataset_train,
                              dataset_x=dataset_semi,
                              num_blocks=3, latent_dim=400,
                              save_dir='asset/semi_bytenet_nltk_comtrans')
model.train(max_ep=50, lr=0.0001)
