
from code.plot.util.tfsummary import TFSummary
from code.plot.util.ggplot import GGPlot
from code.dataset import WMTBilingualNews

import pandas as pd

dataset = WMTBilingualNews(batch_size=64,
                           year=2014,
                           source_lang='de', target_lang='en',
                           min_length=None, max_length=None)

summary = TFSummary(
    'hpc_asset/bytenet_wmt_2014',
    alpha=0.3
)

data = pd.concat(
    [
        summary.read_summary('metrics/model-loss-test_1'),
        summary.read_summary('losses/cross_entropy/supervised-x2y')
    ],
    keys=['test', 'train'],
    names=['dataset']
)

data = data.reset_index(level=['dataset', 'step'])
data['epoch'] = data['step'] / dataset.num_batch

gg = GGPlot("""
p = ggplot(dataframe, aes(x=epoch))
p = p + geom_line(aes(y = value.raw, colour=dataset), alpha=0.2)
p = p + geom_line(aes(y = value.smooth, colour=dataset))
p = p + labs(x="batch epoch", y="cross entropy")
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=5.5, units="cm");
""")

gg.run(data, 'theory/overfitting')
