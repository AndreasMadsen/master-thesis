import argparse

from code.dataset import NLTKComtrans, WMTBilingualNews

parser = argparse.ArgumentParser()
parser.add_argument("-c", "--dataset", type=str, default='wmt',
                    choices=('nltk', 'wmt'))
parser.add_argument("-s", "--source-lang", type=str, default='de')
parser.add_argument("-t", "--target-lang", type=str, default='en')
parser.add_argument("-y", "--year", type=int, default=2014)
parser.add_argument("-p", "--part", type=str, default='source',
                    choices=('source', 'target'))
args = parser.parse_args()

# build dataset
if args.dataset == 'wmt':
    dataset = WMTBilingualNews(
        year=args.year,
        source_lang=args.source_lang, target_lang=args.target_lang
    )
elif args.dataset == 'nltk':
    dataset = NLTKComtrans(
        source_lang=args.source_lang, target_lang=args.target_lang
    )

# print dataset
for sentence in dataset:
    if args.part == 'source':
        print(sentence[0])
    elif args.part == 'target':
        print(sentence[1])
