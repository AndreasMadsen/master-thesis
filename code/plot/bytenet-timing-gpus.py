
from code.plot.util.tfsummary import TFSummary
from code.plot.util.ggplot import GGPlot

import pandas as pd


def observation_speed(dataframe, gpus):
    dataframe['value raw'] = dataframe['value raw'] * 16 * gpus
    dataframe['value smooth'] = dataframe['value smooth'] * 16 * gpus
    return dataframe


summary_gpu1 = TFSummary(
    'hpc_asset/bytenet_wmt_2014_xla_compare_timeing'
    '/bytenet_wmt_2014_gpu1_noxla',
    alpha=0.1
)

summary_gpu4 = TFSummary(
    'hpc_asset/bytenet_wmt_2014_xla_compare_timeing'
    '/bytenet_wmt_2014_gpu4_noxla',
    alpha=0.1
)

speed = pd.concat(
    [
        observation_speed(summary_gpu1.read_summary('global_step/sec'), 1),
        observation_speed(summary_gpu4.read_summary('global_step/sec'), 4)
    ],
    keys=['1 GPU', '4 GPUs'],
    names=['parallelism']
)
speed = speed.reset_index(level=['parallelism', 'sec'])

gg = GGPlot("""
dataframe$time = as.POSIXct(dataframe$sec, origin = "1970-01-01", tz = "UTC")

p = ggplot(dataframe, aes(x=time))
p = p + geom_line(aes(y = value.raw, colour=parallelism), alpha=0.2)
p = p + geom_line(aes(y = value.smooth, colour=parallelism))
p = p + labs(x="duration", y="obs./sec")
p = p + scale_x_datetime(date_labels="%Mmin %Hh")
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=5, units="cm");
""")

gg.run(speed, 'bytenet/timing-gpus')
