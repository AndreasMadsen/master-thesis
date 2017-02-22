
import pandas as pd

from code.dataset import WMTBilingualNews
from code.plot.util.ggplot import GGPlot

dataset = WMTBilingualNews(source_lang='de', target_lang='en', year=2014,
                           min_length=0, max_length=1000)

line_length = []

for i, (source, target) in enumerate(dataset):
    line_length.append((len(source), 'Deutsch'))
    line_length.append((len(target), 'English'))

df = pd.DataFrame(line_length, columns=('length', 'language'))

gg = GGPlot("""
p = ggplot(dataframe, aes(x=length, fill=language))
p = p + geom_density(alpha=0.5)
p = p + theme(text = element_text(size=10))

ggsave(filepath, p, width=13, height=6, units="cm");
""")

gg.run(df, 'theory/wmt-deen-density.pdf')
