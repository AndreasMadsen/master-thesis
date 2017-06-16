
import pandas as pd

from code.plot.util.ggplot import GGPlot
from code.plot.util.result import load_result
from code.plot.util.tfsummary import TFSummary

semi_supervised = load_result('semi-supervised-synthetic-digits-grid')

wall_time = []
for row_i, row in semi_supervised.iterrows():
    model_name = f'train_{int(row["labeled size"])}' \
                 f'_semi_{int(row["unlabled size"])}' \
                 f'_factor_{row["semi-supervised factor"]}'

    summary = TFSummary(
        'hpc_asset/semi_bytenet_synthetic_digits_grid/' + model_name,
        alpha=0.05
    )
    wall_time.append(summary.wall_time())

semi_supervised['wall time'] = pd.Series(
    wall_time, index=semi_supervised.index
)

semi_supervised = semi_supervised.groupby(
    ['labeled size', 'unlabled size'], as_index=False
).mean()

del semi_supervised['misclassification rate']
del semi_supervised['semi-supervised factor']

semi_supervised.to_csv('time-result.csv', index=False)

gg = GGPlot("""
library(latex2exp)
# intrepret wall time as time
dataframe$time = as.POSIXct(dataframe$wall.time, origin = "1970-01-01", tz = "UTC")

size.to.label = function (value) {
    return(TeX(sprintf("labeled obs.: $%s$", value)))
}

p = ggplot(dataframe, aes(x=unlabled.size))
p = p + geom_point(aes(y = time), colour="#00BFC4")
p = p + geom_line(aes(y = time), colour="#00BFC4", linetype="dashed")
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
