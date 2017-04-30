
from code.plot.util.tfsummary import TFSummary
from code.plot.util.ggplot import GGPlot

import pandas as pd

summary = TFSummary(
    'hpc_asset/bytenet_synthetic_digits'
)

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

raw = pd.concat(
  [missrate, entropy],
  keys=['misclassification rate', 'cross entropy'],
  names=['loss type']
)

data = pd.merge(
    raw,
    raw.ewm(alpha=0.25).mean(),
    left_index=True, right_index=True,
    suffixes=(' raw', ' smooth')
)

data = data.reset_index(level=['loss type', 'dataset', 'sec'])

gg = GGPlot("""
dataframe$time = as.POSIXct(dataframe$sec, origin = "1970-01-01", tz = "UTC")

p = ggplot(dataframe, aes(x=time))
p = p + geom_line(aes(y = value.raw, colour=dataset), alpha=0.2)
p = p + geom_line(aes(y = value.smooth, colour=dataset))
p = p + facet_wrap(~loss.type, ncol=1, scales="free_y")
p = p + scale_x_datetime(date_labels="%M min.")
p = p + labs(x="duration", y="")
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=9, units="cm");
""")

gg.run(data, 'bytenet/validation-synthetic-digits')
