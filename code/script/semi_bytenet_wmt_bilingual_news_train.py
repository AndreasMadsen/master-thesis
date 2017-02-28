
import sugartensor as stf

from code.dataset import WMTBilingualNews
from code.model import SemiSupervisedByteNet

# set log level to debug
stf.sg_verbosity(10)


dataset_train = WMTBilingualNews(batch_size=8,
                                 year=2014,
                                 source_lang='de', target_lang='en')

dataset_semi = WMTBilingualNews(batch_size=8,
                                year=2015,
                                source_lang='de', target_lang='en',
                                vocabulary=dataset_train.vocabulary,
                                validate=True)

model = SemiSupervisedByteNet(dataset_train,
                              dataset_x=dataset_semi,
                              num_blocks=3, latent_dim=400,
                              save_dir='asset/semi_bytenet_wmt_deen')
model.train(max_ep=5, lr=0.0001, profile=10)
