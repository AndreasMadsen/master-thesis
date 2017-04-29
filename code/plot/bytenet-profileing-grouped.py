
from code.plot.tracing import Tracing, Group, Aggregate
from code.plot.util.ggplot import GGPlot

trace = Tracing('bytenet_wmt_2014_gpu4_noxla_profile.json')
group = Group(trace)
agg = Aggregate(group)
data = agg.dataframe()

gg = GGPlot("""
p = ggplot(dataframe)
p = p + geom_rect(aes(xmin=start, xmax=start+duration, ymin=level, ymax=level+1, fill=name), color="black")
p = p + facet_grid(device ~ .)
p = p + labs(x="seconds", fill="")
p = p + theme(legend.position="bottom",
              text=element_text(size=10),
              axis.title.y=element_blank(),
              axis.text.y=element_blank(),
              axis.ticks.y=element_blank())

ggsave(filepath, p, width=page.width, height=14, units="cm");
""")

gg.run(data, 'bytenet/profile-grouped')
