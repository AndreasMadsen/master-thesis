
import itertools
import json
import re

import tqdm

from code.moses import Tokenizer, multi_bleu


def add_bleu_score(translation_tuple):
    source = translation_tuple['source']
    target = translation_tuple['target']
    translation = translation_tuple['translation']

    # Tokenize
    target_tokenizer = Tokenizer('en')
    translation_tokenizer = Tokenizer('en')

    with target_tokenizer, translation_tokenizer:
        target_tokenizer.write(target)
        translation_tokenizer.write(translation)

    # Calculate BLEU
    text_result = multi_bleu(
        translate=translation_tokenizer,
        target=target_tokenizer
    )

    text_match = re.match(r'^BLEU = ([0-9.]+),', text_result)
    if (text_match is not None):
        bleu_score = float(text_match.group(1))
    else:
        bleu_score = float(0)

    return {
        'source': source,
        'target': target,
        'translation': translation,
        'bleu': bleu_score
    }


with open('result/translation/wmt-dump.json') as translations, \
     open('result/translation/wmt-bleu.json', 'w') as translations_bleu:
    translations_raw = list(translations)
    translations = []

    for translation_raw in tqdm.tqdm(translations_raw):
        translation = add_bleu_score(json.loads(translation_raw))
        translations.append(translation)
        print(json.dumps(translation), file=translations_bleu)


    for translation_tuple in sorted(
        itertools.islice(translations, 100),
        key=lambda o: o['bleu'],
        reverse=True
    ):

        bleu = translation_tuple['bleu']
        source = translation_tuple['source']
        target = translation_tuple['target']
        translation = translation_tuple['translation']

        print('           BLEU: %f' % (bleu, ))
        print('         source: %s' % (source, ))
        print('         target: %s' % (target, ))
        print('    translation: %s' % (translation, ))
