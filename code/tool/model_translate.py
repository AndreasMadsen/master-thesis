import sys
import argparse
import itertools

import tensorflow as tf

from code.model import ByteNet
from code.dataset import WMTBilingualNews


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    for batch in itertools.zip_longest(*args, fillvalue=None):
        if None in batch:
            batch = batch[:batch.index(None)]
        yield batch


def remove_linebreak(batch):
    return tuple([sentence[:-1] for sentence in batch])


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--dataset", type=str, default='wmt')
parser.add_argument("-s", "--source-lang", type=str, default='de')
parser.add_argument("-t", "--target-lang", type=str, default='en')
parser.add_argument("-y", "--year", type=int, default=2014)
parser.add_argument("-m", "--model", type=str, default='ByteNet')
parser.add_argument("-d", "--save-dir", type=str, default='asset/bytenet')
args = parser.parse_args()

# build training dataset
dataset = WMTBilingualNews(
    year=args.year,
    source_lang=args.source_lang, target_lang=args.target_lang
)

# build bytenet graph
model = ByteNet(dataset, save_dir=args.save_dir)
model.train_model()

# start TensorFlow session
with tf.Session() as sess:
    model.restore(sess)

    # build inference model
    x = tf.placeholder(dtype=tf.int32,
                       shape=(16, dataset.effective_max_length))
    translate = model.inference_model(x, reuse=True)

    # translate stdin
    for source_decoded in grouper(sys.stdin, 16):
        source_encoded = dataset.encode_as_batch(
            remove_linebreak(source_decoded)
        )

        translated_encoded = sess.run(translate, {x: source_encoded})
        translated_decoded = dataset.decode_as_batch(translated_encoded,
                                                     show_eos=False)

        for sentence in translated_decoded:
            print(sentence)
