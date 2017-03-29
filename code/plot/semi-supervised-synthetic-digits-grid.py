
import re
import os
import os.path as path
import itertools

from tqdm import tqdm
import numpy as np
import pandas as pd
import tensorflow as tf

from code.dataset import SyntheticDigits
from code.model import SemiSupervisedByteNet
from code.model.util.asset_dir import asset_dir
from code.plot.util.ggplot import GGPlot

# get saved models
data_dir = path.join(asset_dir(), 'semi_bytenet_synthetic_digits_grid')

saved_models = list(filter(lambda filename: filename[0] != '.',
                           os.listdir(data_dir)))

# prepear train dataset (just for setting the vocabulary)
dataset_train = SyntheticDigits(batch_size=8, examples=1, seed=10, tqdm=False)

# compute misclassification rate
dataset_test = SyntheticDigits(batch_size=8, examples=1024,
                               seed=12, repeat=False, shuffle=False,
                               tqdm=False)

# build graph
model = SemiSupervisedByteNet(
    dataset_train,
    num_blocks=3, latent_dim=20, beam_size=5,
    gpus=2
)
model.inference_model(dataset_train.source)

results = []
saved_models_pbar = tqdm(saved_models, unit='model')
for saved_model_dirname in saved_models_pbar:
    saved_models_pbar.set_description(f'{saved_model_dirname}')

    # get model parameters
    match = re.search('^train_([0-9]+)_semi_([0-9]+)_factor_(0\.[0-9]+)$',
                      saved_model_dirname)

    # set saved model dir
    model.set_save_dir(path.join(data_dir, saved_model_dirname))

    # compute translation and pair with target
    predict = model.predict_from_dataset(dataset_test,
                                         use_queue=False,
                                         show_eos=True, reuse=True)
    predict_pbar = tqdm(predict,
                        total=dataset_test.num_observation,
                        unit='obs', desc='translating')

    # compute misclassification rate
    miss_rate_list = []
    for source, target, translation in predict_pbar:
        match_count = sum(target_char == translation_char
                          for target_char, translation_char
                          in zip(target, translation))
        miss_rate_list.append(
            (len(translation) - match_count) / len(translation)
        )

    # append results
    results.append({
        'labeled size': int(match.group(1)),
        'unlabled size': int(match.group(2)),
        'semi-supervised factor': float(match.group(3)),
        'misclassification rate': float(np.mean(miss_rate_list))
    })


df = pd.DataFrame(results, columns=(
        'labeled size', 'unlabled size',
        'semi-supervised factor', 'misclassification rate'
    )
)


gg = GGPlot("""
# reorder factors in dataframe
dataframe$unlabled.size = factor(dataframe$unlabled.size,
                                 levels=c("0", "512", "1024"), ordered=TRUE)

p = ggplot(dataframe, aes(x=labeled.size))
p = p + geom_point(aes(y = misclassification.rate, colour=unlabled.size),
                   position=position_dodge(width=0.1))
p = p + facet_grid(semi.supervised.factor ~ .)
p = p + scale_x_continuous(trans="log2")
p = p + scale_colour_manual(values=brewer.pal(7, "PuBu")[c(4, 6, 7)])
p = p + labs(x="labled observations",
             y="misclassification rate",
             colour="unlabled observations")
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=11, units="cm");
""")

gg.run(df, 'semi-bytenet/synthetic-digits-grid')
