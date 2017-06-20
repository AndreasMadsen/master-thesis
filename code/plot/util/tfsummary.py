
import os
import os.path as path

import tensorflow as tf
import pandas as pd


class TFSummary:
    def __init__(self, logdir, alpha=0.25, max_value=float('inf')):
        rundirs = [
            dirname for dirname in os.listdir(logdir)
            if dirname[0] != '.' and path.isdir(path.join(logdir, dirname))
        ]
        if len(rundirs) < 1:
            raise IOError(f'too few run dirs, {rundirs}')

        self.tfevent_filepaths = []
        for rundir in sorted(rundirs):
            tf_file = [
                filename for filename in os.listdir(path.join(logdir, rundir))
                if filename[0] != '.'
            ]

            if len(tf_file) < 1:
                raise IOError(f'too many event files, {tf_file}')

            self.tfevent_filepaths.append(
                path.join(logdir, rundir, tf_file[0])
            )

        self.alpha = alpha
        self.max_value = max_value

    def summary_iterator(self):
        prev_wall_time = 0
        file_wall_time_offset = 0

        for file_index, eventfile in enumerate(self.tfevent_filepaths):
            first_event_in_file = True

            for i, e in enumerate(tf.train.summary_iterator(eventfile)):
                if first_event_in_file:
                    file_wall_time_offset = prev_wall_time - e.wall_time
                    first_event_in_file = False

                yield (e.wall_time + file_wall_time_offset, e)

            prev_wall_time = e.wall_time + file_wall_time_offset

    def tags(self):
        tags = set()
        for wall_time, e in self.summary_iterator():
            for v in e.summary.value:
                tags.add(v.tag)
        return frozenset(tags)

    def wall_time(self):
        end_time = -float('inf')

        for wall_time, e in self.summary_iterator():
            for v in e.summary.value:
                if v.tag == 'global_step/sec':
                    end_time = max(end_time, wall_time)

        return end_time

    def read_summary(self, tag):
        data = []

        for wall_time, e in self.summary_iterator():
            for v in e.summary.value:
                if v.tag == tag:
                    data.append({
                        'step': e.step,
                        'wall time': wall_time,
                        'value raw': min(self.max_value, v.simple_value)
                    })

        # construct dataframe
        df = pd.DataFrame(data, columns=('step', 'wall time', 'value raw'))

        # set index
        df['sec'] = df['wall time'] - df['wall time'][0]
        df.set_index(['step', 'sec', 'wall time'], inplace=True)

        # smooth values
        df['value smooth'] = df['value raw'].ewm(alpha=self.alpha).mean()

        return df
