
import pandas as pd
import scipy.stats as st
import numpy as np

from code.plot.util.ggplot import GGPlot
from code.plot.util.result import load_result


def ci(x):
    ymin, ymax = st.t.interval(0.95, x.count()-1, loc=0, scale=st.sem(x))
    return ymax


semi_supervised_raw = load_result('semi-supervised-synthetic-digits-grid')
attention = load_result('attention-synthetic-digits-baseline')

semi_supervised_raw.set_index([
    'labeled size', 'unlabled size',
    'semi-supervised factor', 'iteration'
], inplace=True)
semi_supervised_raw.sort_index(inplace=True)

grouped = semi_supervised_raw.groupby(level=[
    'labeled size', 'unlabled size', 'semi-supervised factor'
])

semi_supervised = grouped.agg(['mean', ci])
semi_supervised = semi_supervised.reset_index(
    level=['labeled size', 'unlabled size', 'semi-supervised factor']
)

attention['ci'] = np.nan
attention.columns = pd.MultiIndex.from_tuples([
    ('labeled size', ''),
    ('misclassification rate', 'mean'), ('misclassification rate', 'ci')
])

attention_merge = pd.merge(
    semi_supervised, attention, on='labeled size', how='left',
    suffixes=(' semi bytenet', ' baseline')
)

del attention_merge[('misclassification rate semi bytenet', 'mean')]
del attention_merge[('misclassification rate semi bytenet', 'ci')]
attention_merge[('misclassification rate', 'mean')] = \
    attention_merge[('misclassification rate baseline', 'mean')]
attention_merge[('misclassification rate', 'ci')] = \
    attention_merge[('misclassification rate baseline', 'ci')]
del attention_merge[('misclassification rate baseline', 'mean')]
del attention_merge[('misclassification rate baseline', 'ci')]

data = pd.concat(
    [
        attention_merge,
        semi_supervised
    ],
    keys=['Attention baseline', 'Semi-supervised ByteNet'],
    names=['model']
)
data = data.reset_index(level=['model'])
data.columns = data.columns.to_series().apply(
    lambda x: x[0] if x[1] == '' else f'{x[0]} {x[1]}'
)

gg = GGPlot("""
library(latex2exp)
# reorder factors in dataframe
col.index = 0;
size.to.label = function (value) {
  col.index <<- col.index + 1;
  if (col.index == 1) {
    return(TeX(sprintf("labeled obs.: $%s$", value)))
  } else {
    return(TeX(sprintf("$\\\\lambda = %s$", value)));
  }
}

p = ggplot(dataframe, aes(x=unlabled.size))
p = p + geom_point(aes(y = misclassification.rate.mean, colour=model))
p = p + geom_errorbar(aes(
        ymin = misclassification.rate.mean - misclassification.rate.ci,
        ymax = misclassification.rate.mean + misclassification.rate.ci,
        colour=model,
        width=128
    ))
p = p + geom_line(aes(y = misclassification.rate.mean, linetype=model, colour=model))
p = p + facet_grid(semi.supervised.factor ~ labeled.size, labeller=as_labeller(size.to.label, default = label_parsed))
p = p + scale_x_continuous(breaks=c(0, 512, 1024))
p = p + labs(x="unlabled observations",
             y="misclassification rate")
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=11, units="cm");
""")

gg.run(data, 'semi-bytenet/synthetic-digits-grid')
