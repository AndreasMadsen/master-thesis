
from code.plot.tracing import Tracing, Group, Aggregate
from code.plot.util.ggplot import GGPlot


def group_and_aggregate(trace):
    group = Group(trace)
    agg = Aggregate(group)
    return agg.dataframe()


trace_gpu1 = Tracing(
    'hpc_asset/bytenet_wmt_2014_profile'
    '/bytenet_wmt_2014_gpu1_profile/timeline.json'
)
trace_gpu4 = Tracing(
    'hpc_asset/bytenet_wmt_2014_profile'
    '/bytenet_wmt_2014_gpu4_profile/timeline.json'
)

gg = GGPlot("""
p = ggplot(dataframe)
p = p + geom_rect(aes(xmin=start, xmax=start+duration, ymin=level, ymax=level+1, fill=name), color="black", alpha=0.7, lwd=0.2)
p = p + facet_grid(device ~ .)
p = p + labs(x="seconds", fill="")
p = p + theme(legend.position="bottom",
              text=element_text(size=10),
              axis.title.y=element_blank(),
              axis.text.y=element_blank(),
              axis.ticks.y=element_blank())
p = p + scale_colour_manual(breaks=c(
    "forward", "backward", "encoder", "decoder",
    "embedding", "pre-normalization", "conv-dilated", "other",
    "recover-dim", "reduce-dim"
))

ggsave(filepath, p, width=page.width, height=height, units="cm");
""")

gg.run(group_and_aggregate(trace_gpu4), 'bytenet/profile-grouped-gpu4',
       height=14)
gg.run(group_and_aggregate(trace_gpu1), 'bytenet/profile-grouped-gpu1',
       height=8)
