
import sugartensor as stf

from code.dataset import NLTKComtrans, WMTBilingualNews
from code.model import ByteNet

# set log level to debug
stf.sg_verbosity(10)

dataset_train = WMTBilingualNews(batch_size=64,
                                 year=2014,
                                 source_lang='de', target_lang='en',
                                 min_length=None, max_length=None)

dataset_test = WMTBilingualNews(batch_size=10,
                                year=2014, source_lang='de', target_lang='en',
                                vocabulary=dataset_train.vocabulary,
                                validate=True,
                                shuffle=False, repeat=False)
model = ByteNet(dataset_train,
                num_blocks=3, latent_dim=400,
                save_dir='hpc_asset/bytenet_wmt_2014_300ep')

translation_tuple = model.predict_from_dataset(dataset_test, samples=10)

for i, (source, target, translation) in zip(range(10), translation_tuple):
    print('  %d  source: %s' % (i, source))
    print('     target: %s' % (target, ))
    print('    predict: %s' % (translation, ))
