
from typing import Optional, Iterator, Tuple, NamedTuple
import xml.dom.minidom
import tarfile

from code.download import WMTCache
from code.dataset.abstract.text_dataset import TextDataset
from code.dataset.util.length_checker import LengthChecker

BilingualPair = NamedTuple('BilingualPair', [
    ('tarball', str),
    ('source_filename', str),
    ('target_filename', str)
])

BilingualTarball = NamedTuple('BilingualTarball', [
    ('name', str),
    ('url', str)
])


_wmt_bilingual_news_tarball = {
    'dev': BilingualTarball(
        'bilingual-news-2016-dev.tgz',
        'http://data.statmt.org/wmt16/translation-task/dev.tgz'
    ),
    'test': BilingualTarball(
        'bilingual-news-2016-test.tgz',
        'http://data.statmt.org/wmt16/translation-task/test.tgz'
    )
}

_wmt_bilingual_news_filename = {
    (2013, 'fr', 'en'): BilingualPair(
        tarball='dev',
        source_filename='dev/newstest2013-src.fr.sgm',
        target_filename='dev/newstest2013-ref.en.sgm'
    ),
    (2013, 'de', 'en'): BilingualPair(
        tarball='dev',
        source_filename='dev/newstest2013-src.de.sgm',
        target_filename='dev/newstest2013-ref.en.sgm'
    ),
    (2014, 'fr', 'en'): BilingualPair(
        tarball='dev',
        source_filename='dev/newstest2014-fren-src.fr.sgm',
        target_filename='dev/newstest2014-fren-ref.en.sgm'
    ),
    (2014, 'de', 'en'): BilingualPair(
        tarball='dev',
        source_filename='dev/newstest2014-deen-src.de.sgm',
        target_filename='dev/newstest2014-deen-ref.en.sgm'
    ),
    (2015, 'de', 'en'): BilingualPair(
        tarball='dev',
        source_filename='dev/newstest2015-deen-src.de.sgm',
        target_filename='dev/newstest2015-deen-ref.en.sgm'
    ),
    (2015, 'ru', 'en'): BilingualPair(
        tarball='dev',
        source_filename='dev/newstest2015-ruen-src.ru.sgm',
        target_filename='dev/newstest2015-ruen-ref.en.sgm'
    )
}

# append swaped target and source
_wmt_bilingual_news_filename.update({
    (key_year, key_target, key_source): BilingualPair(
        val_file, val_target, val_source
    )

    for (key_year, key_source, key_target), (val_file, val_source, val_target)
    in _wmt_bilingual_news_filename.items()
})


class WMTBilingualNews(TextDataset):
    _files: BilingualPair
    _tarball: BilingualTarball
    _length_checker: LengthChecker
    _max_observations: int = None

    def __init__(self,
                 year: int=2013,
                 source_lang: str='fr',
                 target_lang: str='en',
                 min_length: Optional[int]=50, max_length: Optional[int]=150,
                 max_observations=None,
                 **kwargs) -> None:

        self._files = _wmt_bilingual_news_filename[
            (year, source_lang, target_lang)
        ]
        self._tarball = _wmt_bilingual_news_tarball[
            self._files.tarball
        ]

        self._length_checker = LengthChecker(min_length, max_length)
        self._max_observations = max_observations

        super().__init__(
            source_lang, target_lang,
            key=(year, source_lang, target_lang, min_length, max_length),
            name='wmt-bilinual-news',
            **kwargs
        )

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        with WMTCache() as wmt_cache:
            # download tarball
            wmt_cache.download(self._tarball.name, self._tarball.url)

            # peak in tarball
            filepath = wmt_cache.filepath(self._tarball.name)
            with tarfile.open(filepath, 'r:gz', encoding='utf-8') as tar:
                # extract the SGML files from the tarball without unpacking
                # everything
                source_file = tar.extractfile(self._files.source_filename)
                target_file = tar.extractfile(self._files.target_filename)

                # & is perhaps valid in SGML but it is special in XML
                source_text = source_file.read() \
                                         .replace(b'&', b'&amp;') \
                                         .replace(b'<...>', b'&lt;...&gt;')
                target_text = target_file.read() \
                                         .replace(b'&', b'&amp;') \
                                         .replace(b'<...>', b'&lt;...&gt;')

                # parse SGML files as XML
                source_dom = xml.dom.minidom.parseString(source_text)
                target_dom = xml.dom.minidom.parseString(target_text)

                # get all sentences
                source_sentence_elems = source_dom.getElementsByTagName('seg')
                target_sentence_elems = target_dom.getElementsByTagName('seg')

                observations = 0
                for source_sentence_elem, target_sentence_elem in \
                        zip(source_sentence_elems, target_sentence_elems):

                    # assumes the first element is the text node
                    source = source_sentence_elem.firstChild.nodeValue
                    target = target_sentence_elem.firstChild.nodeValue

                    # check string size
                    if self._length_checker(source, target):
                        yield (source, target)
                        observations += 1

                        if self._max_observations is not None and \
                           observations >= self._max_observations:
                            break
