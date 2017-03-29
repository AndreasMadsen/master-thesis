
from typing import List, Optional
import subprocess

from code.moses.util.script_path import script_path


class Tokenizer:
    lang: str
    source: Optional[List[str]]
    tokenized: Optional[List[str]]

    def __init__(self, lang: str) -> None:
        self.lang = lang
        self.source = None
        self.tokenized = None

    def __enter__(self):
        self.source = []
        return self

    def write(self, sequence: str) -> None:
        self.source.append(sequence)

    def __exit__(self, type, value, traceback) -> None:
        result = subprocess.run(
            [
                'perl', script_path('scripts/tokenizer/tokenizer.perl'),
                '-l', self.lang,
                '-no-escape'
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            input='\n'.join(self.source) + '\n',
            encoding='utf-8'
        )

        self.tokenized = [line.rstrip()
                          for line in result.stdout.rstrip().split('\n')]

    def __iter__(self):
        if self.tokenized is None:
            raise ValueError('iter called before finished writeing')

        return iter(self.tokenized)
