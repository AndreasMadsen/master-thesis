
import os
import os.path as path

import tensorflow as tf
import pandas as pd


class TFSummary:
    def __init__(self, logdir, alpha=0.25):
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
        self.alpha = alpha

    def tags(self):
        tags = set()
        for e in tf.train.summary_iterator(self.tfevent_filepath):
            for v in e.summary.value:
                tags.add(v.tag)
        return frozenset(tags)

    def wall_time(self):
        start_time = float('inf')
        end_time = -float('inf')

        for e in tf.train.summary_iterator(self.tfevent_filepath):
            for v in e.summary.value:
                if v.tag == 'global_step/sec':
                    start_time = min(start_time, e.wall_time)
                    end_time = max(end_time, e.wall_time)

        return end_time - start_time

    def read_summary(self, tag):
        data = []

        for e in tf.train.summary_iterator(self.tfevent_filepath):
            for v in e.summary.value:
                if v.tag == tag:
                    data.append({
                        'step': e.step,
                        'wall time': e.wall_time,
                        'value raw': v.simple_value
                    })

        # construct dataframe
        df = pd.DataFrame(data, columns=('step', 'wall time', 'value raw'))

        # set index
        df['sec'] = df['wall time'] - df['wall time'][0]
        df.set_index(['step', 'sec', 'wall time'], inplace=True)

        # smooth values
        df['value smooth'] = df['value raw'].ewm(alpha=self.alpha).mean()

        return df
