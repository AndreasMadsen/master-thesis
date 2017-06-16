
import json

from tqdm import tqdm
import sugartensor as stf

from code.dataset import Europarl, WMTBilingualNews
from code.model import ByteNet
from code.metric import BleuScore, ModelLoss
from code.moses import Tokenizer, multi_bleu

# set log level to debug
stf.sg_verbosity(10)

dataset_train = Europarl(batch_size=64,
                         source_lang='de', target_lang='en',
                         min_length=None, max_length=500,
                         external_encoding='build/europarl-max500.tfrecord')

dataset_test = WMTBilingualNews(batch_size=1,
                                year=2015,
                                source_lang='de', target_lang='en',
                                min_length=None, max_length=None,
                                vocabulary=dataset_train.vocabulary,
                                validate=True,
                                shuffle=False, repeat=False)

model = ByteNet(dataset_train,
                version='v1-small',
                deep_summary=False,
                save_dir='asset/bytenet_small_europarl_max500_adam',
                gpus=1)

# translate
print('predict from dataset:')
result = model.predict_from_dataset(dataset_test, show_eos=False, samples=12)
# show progress
iterator = tqdm(enumerate(result),
                total=dataset_test.num_observation,
                desc="translating", unit='obs',
                dynamic_ncols=True)

# tokenize data
target_tokenizer = Tokenizer('en')
translation_tokenizer = Tokenizer('en')

with open('translation-dump.json', 'w') as translation_dump:
    with target_tokenizer, translation_tokenizer:
        for i, (source, target, translation) in iterator:
            if i < 10:
                tqdm.write(' %d       source: %s' % (i, source))
                tqdm.write('         target: %s' % (target, ))
                tqdm.write('    translation: %s' % (translation, ))

            print(json.dumps({
                "source": source,
                "target": target,
                "translation": translation
            }), file=translation_dump)

            target_tokenizer.write(target)
            translation_tokenizer.write(translation)

# calculate BLEU score
print(multi_bleu(translate=translation_tokenizer, target=target_tokenizer))
