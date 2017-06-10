
from code.plot.util.tfsummary import TFSummary
from code.plot.util.ggplot import GGPlot

import pandas as pd

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

data = data.reset_index(level=['dataset', 'sec'])

gg = GGPlot("""
dataframe$time = as.POSIXct(dataframe$sec, origin = "1970-01-01", tz = "UTC")

p = ggplot(dataframe, aes(x=time))
p = p + geom_line(aes(y = value.raw, colour=dataset), alpha=0.2)
p = p + geom_line(aes(y = value.smooth, colour=dataset))
p = p + labs(x="duration", y="cross entropy")
p = p + scale_x_datetime(date_labels="%Hh")
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=5.5, units="cm");
""")

gg.run(data, 'theory/overfitting')
