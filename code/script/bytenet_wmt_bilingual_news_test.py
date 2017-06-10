
import sugartensor as stf

from code.dataset import NLTKComtrans, WMTBilingualNews
from code.model import ByteNet
from code.moses import Tokenizer, multi_bleu

# set log level to debug
stf.sg_verbosity(10)

dataset_train = WMTBilingualNews(batch_size=64,
                                 year=2014,
                                 source_lang='de', target_lang='en',
                                 min_length=None, max_length=None)

dataset_test = WMTBilingualNews(batch_size=10,
                                year=2015, source_lang='de', target_lang='en',
                                vocabulary=dataset_train.vocabulary,
                                validate=True,
                                shuffle=False, repeat=False)
model = ByteNet(dataset_train,
                num_blocks=3, latent_dim=400,
                save_dir='hpc_asset/bytenet_wmt_2014')

translation_tuple = model.predict_from_dataset(dataset_test, samples=10)

for i, (source, target, translation) in zip(range(10), translation_tuple):
    # Tokenize
    target_tokenizer = Tokenizer('en')
    translation_tokenizer = Tokenizer('en')

    with target_tokenizer, translation_tokenizer:
        target_tokenizer.write(target)
        translation_tokenizer.write(translation)

    # Calculate BLEU
    bleu_result = multi_bleu(
        translate=translation_tokenizer,
        target=target_tokenizer
    )

    print('  %d  source: %s' % (i, source))
    print('     target: %s' % (target, ))
    print('    predict: %s' % (translation, ))
    print('    compare: %s' % (bleu_result, ))
