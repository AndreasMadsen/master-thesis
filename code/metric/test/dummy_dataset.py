
from code.dataset.abstract import TextDataset


class DummyDataset(TextDataset):
    def __init__(self, text, *args, **kwargs):
        self._text = text
        super().__init__(*args, **kwargs)

    def __iter__(self):
        for line in self._text:
            yield ('', line)
