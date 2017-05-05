
import pandas as pd

from code.plot.util.ggplot import GGPlot
from code.plot.util.result import load_result

semi_supervised = load_result('semi-supervised-synthetic-digits-grid')
attention = load_result('attention-synthetic-digits-baseline')

data = pd.merge(
    semi_supervised, attention, on='labeled size', how='left',
    suffixes=(' semi bytenet', ' baseline')
)

gg = GGPlot("""
library(latex2exp)
# reorder factors in dataframe
dataframe$unlabled.size = factor(dataframe$unlabled.size,
                                 levels=c("0", "512", "1024"), ordered=TRUE)

size.to.label = function (value) {
    return(TeX(sprintf("$\\\\lambda = %s$", value)));
}

p = ggplot(dataframe, aes(x=labeled.size))
p = p + geom_point(aes(y = misclassification.rate.semi.bytenet, colour=unlabled.size),
                   position=position_dodge(width=0.1))
p = p + geom_errorbar(aes(ymax=misclassification.rate.baseline, ymin=misclassification.rate.baseline), linetype="dashed", width=0.1)
p = p + facet_grid(semi.supervised.factor ~ ., labeller=as_labeller(size.to.label, default = label_parsed))
p = p + scale_x_continuous(trans="log2")
p = p + scale_colour_manual(values=brewer.pal(7, "PuBu")[c(4, 6, 7)])
p = p + labs(x="labled observations",
             y="misclassification rate",
             colour="unlabled observations")
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=11, units="cm");
""")

gg.run(data, 'semi-bytenet/synthetic-digits-grid')
