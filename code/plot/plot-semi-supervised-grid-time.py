
import pandas as pd
import scipy.stats as st
import numpy as np

from code.plot.util.ggplot import GGPlot
from code.plot.util.result import load_result
from code.plot.util.tfsummary import TFSummary


def ci(x):
    ymin, ymax = st.t.interval(0.95, x.count()-1, loc=0, scale=st.sem(x))
    return ymax


semi_supervised_raw = load_result('semi-supervised-synthetic-digits-grid-walltime')
semi_supervised_raw.set_index([
    'labeled size', 'unlabled size',
    'semi-supervised factor', 'iteration'
], inplace=True)

grouped = semi_supervised_raw.groupby(level=[
    'labeled size', 'unlabled size'
])
semi_supervised = grouped.agg(['mean', ci])

semi_supervised.columns = semi_supervised.columns.to_series().apply(
    lambda x: x[0] if x[1] == '' else f'{x[0]} {x[1]}'
)
semi_supervised = semi_supervised.reset_index(
    level=['labeled size', 'unlabled size']
)

gg = GGPlot("""
library(latex2exp)
# intrepret wall time as time
dataframe$wall.time.mean = as.POSIXct(
    dataframe$wall.time.mean,
    origin = "1970-01-01", tz = "UTC"
)
dataframe$wall.time.ymin = as.POSIXct(
    dataframe$wall.time.mean - dataframe$wall.time.ci,
    origin = "1970-01-01", tz = "UTC"
)
dataframe$wall.time.ymax = as.POSIXct(
    dataframe$wall.time.mean + dataframe$wall.time.ci,
    origin = "1970-01-01", tz = "UTC"
)

size.to.label = function (value) {
    return(TeX(sprintf("labeled obs.: $%s$", value)))
}

p = ggplot(dataframe, aes(x=unlabled.size))
p = p + geom_point(aes(y = wall.time.mean), colour="#00BFC4")
p = p + geom_errorbar(aes(
        ymin = wall.time.ymin,
        ymax = wall.time.ymax
    ),
    width=128,
    colour="#00BFC4")
p = p + geom_line(aes(y = wall.time.mean), colour="#00BFC4", linetype="dashed")
p = p + facet_grid(~ labeled.size, labeller=as_labeller(size.to.label, default = label_parsed))
p = p + scale_y_datetime(date_labels="%Hh")
p = p + scale_x_continuous(limits=c(-128, 1152), breaks=c(0, 512, 1024))
p = p + labs(x="unlabled observations",
             y="wall time")
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=5.2, units="cm");
""")

gg.run(semi_supervised, 'semi-bytenet/synthetic-digits-grid-time')
