
from code.plot.util.tfsummary import TFSummary
from code.plot.util.ggplot import GGPlot
from code.dataset import SyntheticDigits

import pandas as pd

dataset = SyntheticDigits(batch_size=16, examples=128, seed=10)

summary = TFSummary(
    'hpc_asset/bytenet_small_synthetic_digits',
    alpha=0.2
)

print(summary.read_summary('metrics/model-loss-test_1'))

entropy = pd.concat(
    [
        summary.read_summary('metrics/model-loss-test_1'),
        summary.read_summary('losses/cross_entropy/supervised-x2y')
    ],
    keys=['test', 'train'],
    names=['dataset']
)

missrate = pd.concat(
    [
        summary.read_summary('metrics/missrate-test_1'),
        summary.read_summary('metrics/missrate-train_1')
    ],
    keys=['test', 'train'],
    names=['dataset']
)

data = pd.concat(
  [missrate, entropy],
  keys=['misclassification rate', 'cross entropy'],
  names=['loss type']
)
data = data.reset_index(level=['loss type', 'dataset', 'step'])
data['epoch'] = data['step'] / dataset.num_batch

gg = GGPlot("""
p = ggplot(dataframe, aes(x=epoch))
p = p + geom_line(aes(y = value.raw, colour=dataset), alpha=0.2)
p = p + geom_line(aes(y = value.smooth, colour=dataset))
p = p + facet_wrap(~loss.type, ncol=1, scales="free_y")
p = p + labs(x="batch epoch", y="")
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=9, units="cm");
""")

gg.run(data, 'bytenet-small/validation-synthetic-digits')
