
from code.plot.tracing import Tracing, Group, Aggregate
from code.plot.util.ggplot import GGPlot

trace = Tracing('bytenet_wmt_2014_gpu4_noxla_profile.json')
data = trace.dataframe()

gg = GGPlot("""
p = ggplot(dataframe)
p = p + geom_rect(aes(xmin=start, xmax=start+max(duration, 0.01), ymin=thread+0.5, ymax=thread+1.5, fill=name))
p = p + facet_grid(device ~ .)
p = p + labs(x="seconds", y="thread", fill="")
p = p + ylim(0.5, 4.5)
p = p + theme(legend.position="none",
              text=element_text(size=10))

ggsave(filepath, p, width=page.width, height=14, units="cm");
""")

gg.run(data, 'bytenet/profile-raw', file_format='pdf')
