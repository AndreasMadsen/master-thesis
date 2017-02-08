
from typing import Iterator, Tuple, NamedTuple
import xml.dom.minidom
import tarfile

from code.dataset.util.wmt_env import WMTEnv
from code.dataset.abstract.text_dataset import TextDataset

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
        source_filename='dev/newstest2013-ref.fr.sgm',
        target_filename='dev/newstest2013-ref.en.sgm'
    ),
    (2013, 'de', 'en'): BilingualPair(
        tarball='dev',
        source_filename='dev/newstest2013-ref.de.sgm',
        target_filename='dev/newstest2013-ref.en.sgm'
    ),
    (2014, 'fr', 'en'): BilingualPair(
        tarball='dev',
        source_filename='dev/newstest2014-fren-ref.fr.sgm',
        target_filename='dev/newstest2014-fren-ref.en.sgm'
    ),
    (2014, 'de', 'en'): BilingualPair(
        tarball='dev',
        source_filename='dev/newstest2014-deen-ref.de.sgm',
        target_filename='dev/newstest2014-deen-ref.en.sgm'
    ),
    (2015, 'de', 'en'): BilingualPair(
        tarball='dev',
        source_filename='dev/newstest2015-ende-ref.de.sgm',
        target_filename='dev/newstest2015-deen-ref.en.sgm'
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
    _min_length: int
    _max_length: int

    def __init__(self,
                 year: int=2013,
                 source_lang: str='fr',
                 target_lang: str='en',
                 min_length: int=50, max_length: int=150,
                 **kwargs) -> None:

        self._files = _wmt_bilingual_news_filename[
            (year, source_lang, target_lang)
        ]
        self._tarball = _wmt_bilingual_news_tarball[
            self._files.tarball
        ]

        self._min_length = min_length
        self._max_length = max_length

        super().__init__(
            max_length=self._max_length,
            **kwargs
        )

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        with WMTEnv() as wmt_env:
            # download tarball
            wmt_env.download(self._tarball.name, self._tarball.url)

            # peak in tarball
            with tarfile.open(wmt_env.filepath(self._tarball.name), 'r:gz') \
                    as tar:
                # extract the SGML files from the tarball without unpacking
                # everything
                source_file = tar.extractfile(self._files.source_filename)
                target_file = tar.extractfile(self._files.target_filename)

                # & is perhaps valid in SGML but it is special in XML
                source_text = source_file.read().replace(b'&', b'&amp;')
                target_text = target_file.read().replace(b'&', b'&amp;')

                # parse SGML files as XML
                source_dom = xml.dom.minidom.parseString(source_text)
                target_dom = xml.dom.minidom.parseString(target_text)

                # get all sentences
                source_sentence_elems = source_dom.getElementsByTagName('seg')
                target_sentence_elems = target_dom.getElementsByTagName('seg')

                for source_sentence_elem, target_sentence_elem in \
                        zip(source_sentence_elems, target_sentence_elems):

                    # assumes the first element is the text node
                    source = source_sentence_elem.firstChild.nodeValue
                    target = target_sentence_elem.firstChild.nodeValue

                    # check string size
                    if self._min_length <= len(source) < self._max_length and \
                       self._min_length <= len(target) < self._max_length:
                        yield (source, target)
