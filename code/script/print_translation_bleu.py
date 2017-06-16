
import itertools
import json
import re

import tqdm

from code.moses import Tokenizer, multi_bleu

# tokenize data
target_tokenizer = Tokenizer('en')
translation_tokenizer = Tokenizer('en')

with open('result/bytenet-translation/wmt-dump.json') as translations:
    with target_tokenizer, translation_tokenizer:

        for translation_raw in tqdm.tqdm(list(translations)):
            translation_tuple = json.loads(translation_raw)

            target_tokenizer.write(translation_tuple['target'])
            translation_tokenizer.write(translation_tuple['translation'])

# calculate BLEU score
print(multi_bleu(translate=translation_tokenizer, target=target_tokenizer))
