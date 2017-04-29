
import sys
import io
import os
import os.path as path
import tempfile
import subprocess
import textwrap

import pandas as pd

thisdir = path.dirname(path.realpath(__file__))
graphicsdir = path.realpath(
    path.join(thisdir, '..', '..', '..', 'report', 'graphics')
)


class GGPlot:
    r_script: str

    def __init__(self, r_script: str):
        self.r_script = textwrap.dedent(f"""\
        #!/usr/bin/env Rscript
        library(ggplot2);
        library(RColorBrewer);
        page.width = 12.96703;

        args = commandArgs(TRUE);
        dataframe = read.csv(file('stdin'));
        filepath = args[1];

        {r_script}
        """)

    def run(self, dataframe: pd.DataFrame, filepath: str, file_format='pdf'):
        # generate PDF file
        pdf_filepath = path.join(graphicsdir, filepath + '.' + file_format)

        csv_file = io.StringIO()
        dataframe.to_csv(csv_file, index=False)

        with tempfile.NamedTemporaryFile(mode='w') as fd:
            print(self.r_script, file=fd, flush=True)

            env = os.environ.copy()
            env['LANGUAGE'] = 'en'

            try:
                subprocess.run(
                    ['Rscript', fd.name, pdf_filepath],
                    env=env,
                    input=bytes(csv_file.getvalue(), 'utf8'),
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    check=True
                )
            except subprocess.CalledProcessError as e:
                print('Captured R error:', file=sys.stderr)
                print(str(e.stdout, 'utf-8'), file=sys.stderr)
