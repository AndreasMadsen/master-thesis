
from typing import List, Optional
import subprocess

from code.moses.util.script_path import script_path


class Tokenizer:
    lang: str
    process: subprocess.Popen
    tokenized: Optional[List[str]]

    def __init__(self, lang: str) -> None:
        self.lang = lang
        self.tokenized = None

    def __enter__(self):
        self.process = subprocess.Popen(
            [
                'perl', script_path('scripts/tokenizer/tokenizer.perl'),
                '-l', self.lang,
                '-b',  # disable perl buffering, required for streaming
                '-no-escape'  # no HTML escaping of apostrophy, quotes, etc
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            encoding='utf-8'
        )

        return self

    def write(self, sequence: str) -> None:
        self.process.stdin.write(sequence + '\n')

    def __exit__(self, type, value, traceback) -> None:
        self.process.stdin.close()
        self.tokenized = [line.rstrip() for line in self.process.stdout]

    def __iter__(self):
        if self.tokenized is None:
            raise ValueError('iter called before finished writeing')

        return iter(self.tokenized)
