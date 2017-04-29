
import os
import os.path as path

import tensorflow as tf
import pandas as pd


class TFSummary:
    def __init__(self, logdir):
        rundirs = [
            dirname for dirname in os.listdir(logdir)
            if dirname[0] != '.' and path.isdir(path.join(logdir, dirname))
        ]
        if len(rundirs) != 1:
            raise IOError(f'too many run dirs, {rundirs}')

        tfevent_files = [
            filename for filename in os.listdir(path.join(logdir, rundirs[0]))
            if filename[0] != '.'
        ]
        if len(tfevent_files) != 1:
            raise IOError(f'too many event files, {tfevent_files}')

        self.tfevent_filepath = path.join(logdir, rundirs[0], tfevent_files[0])

    def tags(self):
        tags = set()
        for e in tf.train.summary_iterator(self.tfevent_filepath):
            for v in e.summary.value:
                tags.add(v.tag)
        return frozenset(tags)

    def read_summary(self, tag):
        data = []

        for e in tf.train.summary_iterator(self.tfevent_filepath):
            for v in e.summary.value:
                if v.tag == tag:
                    data.append({
                        'step': e.step,
                        'wall time': e.wall_time,
                        'value': v.simple_value
                    })

        df = pd.DataFrame(data, columns=('step', 'wall time', 'value'))
        df['sec'] = df['wall time'] - df['wall time'][0]

        df.set_index(['step', 'sec', 'wall time'], inplace=True)

        return df
