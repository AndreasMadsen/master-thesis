import argparse

from code.dataset import WMTBilingualNews

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--source-lang", type=str, default='de')
parser.add_argument("-t", "--target-lang", type=str, default='en')
parser.add_argument("-y", "--year", type=int, default=2014)
parser.add_argument("-p", "--part", type=str, default='source',
                    choices=('source', 'target'))
args = parser.parse_args()

# build training dataset
dataset = WMTBilingualNews(
    year=args.year,
    source_lang=args.source_lang, target_lang=args.target_lang
)

for sentence in dataset:
    if args.part == 'source':
        print(sentence[0])
    else:
        print(sentence[1])
