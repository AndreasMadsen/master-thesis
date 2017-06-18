
from code.plot.util.tfsummary import TFSummary
from code.plot.util.ggplot import GGPlot

import pandas as pd


def observation_speed(dataframe, gpus):
    dataframe['value raw'] = dataframe['value raw'] * 16 * gpus
    dataframe['value smooth'] = dataframe['value smooth'] * 16 * gpus
    return dataframe


full_gpu1 = TFSummary(
    'hpc_asset/bytenet_wmt_2014_timeing/bytenet_wmt_2014_gpu1_timeing',
    alpha=0.1
)

full_gpu4 = TFSummary(
    'hpc_asset/bytenet_wmt_2014_timeing/bytenet_wmt_2014_gpu4_timeing',
    alpha=0.1
)

small_gpu1 = TFSummary(
    'hpc_asset/bytenet_small_wmt_2014_timeing/bytenet_wmt_2014_gpu1_timeing',
    alpha=0.1
)

small_gpu4 = TFSummary(
    'hpc_asset/bytenet_small_wmt_2014_timeing/bytenet_wmt_2014_gpu4_timeing',
    alpha=0.1
)

selu_gpu4 = TFSummary(
    'hpc_asset/bytenet_selu_wmt_2014_timeing/bytenet_wmt_2014_gpu4_timeing',
    alpha=0.1
)

speed_full = pd.concat(
    [
        observation_speed(full_gpu1.read_summary('global_step/sec'), 1),
        observation_speed(full_gpu4.read_summary('global_step/sec'), 4)
    ],
    keys=['1 GPU', '4 GPUs'],
    names=['parallelism']
)
speed_small = pd.concat(
    [
        observation_speed(small_gpu1.read_summary('global_step/sec'), 1),
        observation_speed(small_gpu4.read_summary('global_step/sec'), 4)
    ],
    keys=['1 GPU', '4 GPUs'],
    names=['parallelism']
)
speed_selu = pd.concat(
    [
        observation_speed(selu_gpu4.read_summary('global_step/sec'), 4)
    ],
    keys=['4 GPUs'],
    names=['parallelism']
)
speed = pd.concat(
    [speed_full, speed_small, speed_selu],
    keys=['bytenet full', 'bytenet small', 'bytenet selu'],
    names=['type']
)
speed = speed.reset_index(level=['type', 'parallelism', 'sec'])

gg = GGPlot("""
dataframe$time = as.POSIXct(dataframe$sec, origin = "1970-01-01", tz = "UTC")

p = ggplot(dataframe, aes(x=time))
p = p + geom_line(aes(y = value.raw, colour=parallelism, linetype=type), alpha=0.2)
p = p + geom_line(aes(y = value.smooth, colour=parallelism, linetype=type))
p = p + labs(x="duration", y="obs./sec")
p = p + scale_x_datetime(date_labels="%Hh")
p = p + theme(legend.position="bottom",
              legend.direction="horizontal",
              legend.box="vertical",
              legend.margin=margin(t=0, unit='cm'),
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=7, units="cm");
""")

gg.run(speed, 'bytenet-selu/timing-gpus')
