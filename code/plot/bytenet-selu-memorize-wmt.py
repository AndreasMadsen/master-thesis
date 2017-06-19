
from code.plot.util.tfsummary import TFSummary
from code.plot.util.ggplot import GGPlot
from code.dataset import WMTBilingualNews

import pandas as pd
import numpy as np

dataset = WMTBilingualNews(batch_size=64,
                           year=2014,
                           source_lang='de', target_lang='en',
                           min_length=None, max_length=None)


def model_dataframe(modelpath, alpha=0.1):
    summary = TFSummary(
        modelpath,
        alpha=alpha
    )

    entropy = pd.concat(
        [
            summary.read_summary('metrics/model-loss-test_1'),
            summary.read_summary('losses/cross_entropy/supervised-x2y')
        ],
        keys=['test', 'train'],
        names=['dataset']
    )

    bleu = pd.concat(
        [
            summary.read_summary('metrics/BLEU-score-test_1'),
            summary.read_summary('metrics/BLEU-score-train_1')
        ],
        keys=['test', 'train'],
        names=['dataset']
    )

    data = pd.concat(
      [bleu, entropy],
      keys=['BLEU score', 'cross entropy'],
      names=['loss type']
    )
    return data


selu_model = model_dataframe(
    'hpc_asset/bytenet_selu_wmt_2014_timeing/bytenet_wmt_2014_gpu4_timeing',
    alpha=0.3
)

normal_model = model_dataframe(
    'hpc_asset/bytenet_wmt_2014',
    alpha=0.3
)
normal_model['value raw'] = np.nan

data = pd.concat(
  [selu_model, normal_model],
  keys=['SELU ByteNet', 'ByteNet'],
  names=['type']
)

data = data.reset_index(level=['type', 'loss type', 'dataset', 'step'])
data['epoch'] = data['step'] / dataset.num_batch
data = data.loc[data['epoch'] <= 150]

gg = GGPlot("""
p = ggplot(dataframe, aes(x=epoch))
p = p + geom_line(aes(y = value.raw, colour=dataset, linetype=type), alpha=0.2)
p = p + geom_line(aes(y = value.smooth, colour=dataset, linetype=type))
p = p + facet_wrap(~loss.type, ncol=1, scales="free_y")
p = p + labs(x="batch epoch", y="")
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=9, units="cm");
""")

gg.run(data, 'bytenet-selu/validation-memorize-wmt')
