import sys
import argparse
import itertools

import tensorflow as tf
import tqdm

from code.model import ByteNet
from code.tool.util.get_dataset import get_dataset


parser = argparse.ArgumentParser()
parser.add_argument("--source-lang", type=str, default='de')
parser.add_argument("--target-lang", type=str, default='en')

parser.add_argument("--train-dataset", type=str, default='wmt',
                    choices=('nltk', 'wmt'))
parser.add_argument("--train-year", type=int, default=2014)

parser.add_argument("--test-dataset", type=str, default='wmt',
                    choices=('nltk', 'wmt'))
parser.add_argument("--test-year", type=int, default=2015)

parser.add_argument("--model", type=str, default='ByteNet')
parser.add_argument("--save-dir", type=str, default='asset/bytenet')

parser.add_argument("--source-file", type=str, default='source.de.txt')
parser.add_argument("--target-file", type=str, default='target.en.txt')
parser.add_argument("--translate-file", type=str, default='translate.en.txt')

args = parser.parse_args()

print(f'Translation {args.source_lang} -> {args.target_lang}')

# build training dataset
train_dataset = get_dataset(dataset=args.train_dataset,
                            year=args.train_year,
                            source_lang=args.source_lang,
                            target_lang=args.target_lang)

test_dataset = get_dataset(dataset=args.test_dataset,
                           year=args.test_year,
                           source_lang=args.source_lang,
                           target_lang=args.target_lang,
                           vocabulary=train_dataset.vocabulary,
                           validate=True,
                           shuffle=False, repeat=False,
                           batch_size=16)

# build bytenet graph
model = ByteNet(train_dataset, save_dir=args.save_dir)

# start translation
translation_tuple = model.predict_from_dataset(test_dataset, show_eos=False)

print(f'starting translation ...')

with open(args.source_file, 'w') as source_fd, \
     open(args.target_file, 'w') as target_fd, \
     open(args.translate_file, 'w') as translate_fd:

    pbar = tqdm.trange(test_dataset.num_observation,
                       desc="translating", unit='sent',
                       dynamic_ncols=True)
    for i, (source, target, translation) in zip(pbar, translation_tuple):
        if i < 10:
            pbar.write(' %d       source: %s' % (i, source))
            pbar.write('         target: %s' % (target, ))
            pbar.write('    translation: %s' % (translation, ))

        print(source, file=source_fd)
        print(target, file=target_fd)
        print(translation, file=translate_fd)
