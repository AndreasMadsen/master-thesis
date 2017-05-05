
import sys
import io
import os
import os.path as path
import json
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
    r_template: str

    def __init__(self, r_script: str):
        self.r_script = r_script
        self.r_template = textwrap.dedent("""\
        #!/usr/bin/env Rscript
        library(ggplot2);
        library(RColorBrewer);
        page.width = 12.96703;

        args = commandArgs(TRUE);
        dataframe = read.csv(file('stdin'));
        filepath = args[1];
        {external_arguments}

        {r_script}
        """)

    def run(self, dataframe: pd.DataFrame, filepath: str, file_format='pdf',
            **kwargs):
        # generate PDF file
        pdf_filepath = path.join(graphicsdir, filepath + '.' + file_format)

        csv_file = io.StringIO()
        dataframe.to_csv(csv_file, index=False)

        with tempfile.NamedTemporaryFile(mode='w') as fd:
            print(self.r_template.format(
                external_arguments='\n'.join([
                    f"{name.replace('_', '.')} = {json.dumps(value)};"
                    for name, value in kwargs.items()
                ]),
                r_script=self.r_script
            ), file=fd, flush=True)

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
