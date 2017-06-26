
import re
import os
import os.path as path
import itertools

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf

from code.model.util.asset_dir import asset_dir
from code.plot.util.result import save_result
from code.plot.util.tfsummary import TFSummary

# get saved models
data_dir = path.join(asset_dir(), 'semi_bytenet_synthetic_digits_grid_stats')

saved_models = list(filter(lambda filename: filename[0] != '.',
                           os.listdir(data_dir)))

results = []
saved_models_pbar = tqdm(saved_models, unit='model')
for saved_model_dirname in saved_models_pbar:
    saved_models_pbar.set_description(f'{saved_model_dirname}')

    # get model parameters
    match = re.search('^train_([0-9]+)_semi_([0-9]+)_factor_(0\.[0-9]+)'
                      '_iter([0-9]+)$',
                      saved_model_dirname)

    summary = TFSummary(
        path.join(data_dir, saved_model_dirname),
        alpha=0.05
    )

    # append results
    results.append({
        'labeled size': int(match.group(1)),
        'unlabled size': int(match.group(2)),
        'semi-supervised factor': float(match.group(3)),
        'iteration': int(match.group(4)),
        'wall time': float(summary.wall_time())
    })


df = pd.DataFrame(results, columns=(
        'labeled size', 'unlabled size',
        'semi-supervised factor', 'iteration',
        'wall time'
    )
)

print(df)

save_result(df, 'semi-supervised-synthetic-digits-grid-walltime')
