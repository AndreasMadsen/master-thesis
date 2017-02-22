
from code.dataset import NLTKComtrans, WMTBilingualNews


def get_dataset(dataset, source_lang, target_lang, year, **kwargs):
    # build training dataset
    if dataset == 'wmt':
        return WMTBilingualNews(
            year=year,
            source_lang=source_lang, target_lang=target_lang,
            **kwargs
        )
    elif dataset == 'nltk':
        return NLTKComtrans(
            source_lang=source_lang, target_lang=target_lang,
            **kwargs
        )
