from code.dataset.europarl import Europarl
from code.plot.util.tfsummary import TFSummary
from code.plot.util.ggplot import GGPlot

import pandas as pd
import numpy as np

dataset = Europarl(batch_size=64,
                   source_lang='de', target_lang='en',
                   min_length=None, max_length=500,
                   external_encoding='build/europarl-max500.tfrecord')


def model_dataframe(filepath):
    summary = TFSummary(
        filepath,
        alpha=0.05
    )

    entropy = pd.concat(
        [
            summary.read_summary('metrics/model-loss-test_1'),
            summary.read_summary('losses/cross_entropy/supervised-x2y')
        ],
        keys=['test', 'train'],
        names=['dataset']
    )

    bleu_test = pd.concat(
        [
            summary.read_summary('metrics/BLEU-score-test_1')
        ],
        keys=['test'],
        names=['dataset']
    )

    bleu_train = pd.concat(
        [
            summary.read_summary('metrics/BLEU-score-train_1')
        ],
        keys=['train'],
        names=['dataset']
    )

    data = pd.concat(
      [bleu_test, bleu_train, entropy],
      keys=['BLEU\u00A0score', 'BLEU score', 'cross entropy'],
      names=['loss type']
    )

    batch_lines = summary.read_summary('losses/cross_entropy/supervised-x2y')
    batch_lines = batch_lines.reset_index(level=['sec', 'step'])
    batch_lines = batch_lines.loc[
        np.diff(batch_lines['step'] % dataset.num_batch) <= 0, :
    ]
    batch_lines = batch_lines['sec'].tolist()

    return data, batch_lines


bytenet_full, _ = model_dataframe(
    'hpc_asset/bytenet_europarl_nosummary_max500_adam'
)
bytenet_full['value raw'] = np.nan

bytenet_small, batch_lines = model_dataframe(
    'hpc_asset/bytenet_small_europarl_max500_adam'
)

data = pd.concat(
  [bytenet_full, bytenet_small],
  keys=['bytenet full', 'bytenet small'],
  names=['type']
)

data = data.reset_index(level=['type', 'loss type', 'dataset', 'sec', 'step'])
data.to_csv('file.csv', index=False)

gg = GGPlot(f"""
library(lubridate);
dataframe$time = as.POSIXct(dataframe$sec, origin = "1970-01-01", tz = "UTC")

p = ggplot(dataframe, aes(x=time))
p = p + geom_line(aes(y = value.raw, colour=dataset, linetype=type), na.rm=TRUE, alpha=0.2)
p = p + geom_line(aes(y = value.smooth, colour=dataset, linetype=type), size=0.5)
p = p + facet_wrap(~loss.type, ncol=1, scales="free_y")
p = p + labs(x="duration", y="")
p = p + scale_x_datetime(labels=function (time) {{
          return(sprintf("%.0fd %.0fh", yday(time) - 1, hour(time)));
        }})
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=12, units="cm");
""")

gg.run(data, 'bytenet-small/europarl')
