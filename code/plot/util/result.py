
import os
import os.path as path

import pandas as pd

thisdir = path.dirname(path.realpath(__file__))
resultdir = path.realpath(
    path.join(thisdir, '..', '..', '..', 'result', 'plot')
)


def save_result(dataframe: pd.DataFrame, filepath: str) -> None:
    # save datafile
    csv_filepath = path.join(resultdir, filepath + '.csv')
    os.makedirs(path.dirname(csv_filepath), exist_ok=True)
    dataframe.to_csv(csv_filepath, index=False)


def load_result(filepath: str) -> pd.DataFrame:
    # save datafile
    csv_filepath = path.join(resultdir, filepath + '.csv')
    return pd.read_csv(csv_filepath, index_col=False)
