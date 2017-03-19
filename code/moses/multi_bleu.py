
from typing import List
import tempfile
import subprocess

from code.moses.util.script_path import script_path


def multi_bleu(translate: List[str], target: List[str]) -> str:
    with tempfile.NamedTemporaryFile('w', encoding='utf-8') as fd:
        for line in target:
            print(line, file=fd, flush=True)

        result = subprocess.run(
            [
                'perl', script_path('scripts/generic/multi-bleu.perl'),
                fd.name
            ],
            encoding='utf-8',
            input='\n'.join(translate) + '\n',
            stderr=subprocess.STDOUT,
            stdout=subprocess.PIPE
        )

        return result.stdout.rstrip()
