
from code.plot.tracing import Tracing, Group, Aggregate
from code.plot.util.ggplot import GGPlot

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
p = p + geom_rect(aes(xmin=start, xmax=start+duration, ymin=thread+0.5, ymax=thread+1.5, fill=name))
p = p + facet_grid(device ~ .)
p = p + labs(x="seconds", y="thread", fill="")
p = p + ylim(0.5, 4.5)
p = p + theme(legend.position="none",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=height, units="cm");
""")

gg.run(trace_gpu4.dataframe(), 'bytenet/profile-raw-gpu4', file_format='pdf',
       height=13)
gg.run(trace_gpu1.dataframe(), 'bytenet/profile-raw-gpu1', file_format='pdf',
       height=7)
