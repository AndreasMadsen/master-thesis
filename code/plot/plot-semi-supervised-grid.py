
import pandas as pd

from code.plot.util.ggplot import GGPlot
from code.plot.util.result import load_result

semi_supervised = load_result('semi-supervised-synthetic-digits-grid')
attention = load_result('attention-synthetic-digits-baseline')

attention_merge = pd.merge(
    semi_supervised, attention, on='labeled size', how='left',
    suffixes=(' semi bytenet', ' baseline')
)
attention_merge['misclassification rate'] = \
    attention_merge['misclassification rate baseline']
del attention_merge['misclassification rate semi bytenet']
del attention_merge['misclassification rate baseline']


data = pd.concat(
    [
        attention_merge,
        semi_supervised
    ],
    keys=['Attention baseline', 'Semi-supervised ByteNet'],
    names=['model']
)
data = data.reset_index(level=['model'])

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
p = p + geom_point(aes(y = misclassification.rate, colour=model))
p = p + geom_line(aes(y = misclassification.rate, linetype=model, colour=model))
p = p + facet_grid(semi.supervised.factor ~ labeled.size, labeller=as_labeller(size.to.label, default = label_parsed))
p = p + scale_x_continuous(limits=c(-128, 1152), breaks=c(0, 512, 1024))
p = p + labs(x="unlabled observations",
             y="misclassification rate")
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=11, units="cm");
""")

gg.run(data, 'semi-bytenet/synthetic-digits-grid')
