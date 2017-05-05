
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

gg = GGPlot("""
library(latex2exp)
# reorder factors in dataframe
dataframe$unlabled.size = factor(dataframe$unlabled.size,
                                 levels=c("0", "512", "1024"), ordered=TRUE)
# intrepret wall time as time
dataframe$time = as.POSIXct(dataframe$wall.time, origin = "1970-01-01", tz = "UTC")

size.to.label = function (value) {
    return(TeX(sprintf("$\\\\lambda = %s$", value)));
}

p = ggplot(dataframe, aes(x=labeled.size))
p = p + geom_point(aes(y = time, colour=unlabled.size),
                   position=position_dodge(width=0.1))
p = p + scale_y_datetime(date_labels="%Hh")
p = p + scale_x_continuous(trans="log2")
p = p + scale_colour_manual(values=brewer.pal(7, "PuBu")[c(4, 6, 7)])
p = p + labs(x="labled observations",
             y="wall time",
             colour="unlabled observations")
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=5.2, units="cm");
""")

gg.run(semi_supervised, 'semi-bytenet/synthetic-digits-grid-time')
