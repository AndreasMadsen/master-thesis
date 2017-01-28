
library(ggplot2)

sigmoid = function (x) {
  return(1/(1 + exp(-x)))
}

sigmoid.inv = function (y) {
  return(-log(1/y - 1))
}

sigmoid.inv.diff = function (y) {
  return(-1/(y*(y - 1)))
}

sigmoid.dist = function (y, mean, sd) {
  return (dnorm(sigmoid.inv(y), mean=mean, sd=sd) * sigmoid.inv.diff(y))
}

settings = data.frame(mu=c(0, 1, 0), mean=c(1, 1, 2))

dat = data.frame(x=c(), density=c(), labels=c(), name=c())
each.row = function (row) {
  label = paste0('N(mu == ', row[1] ,', sigma == ', row[2] ,')')
  dat <<- rbind(dat, data.frame(
    x=seq(-2, 2, 0.01),
    density=dnorm(seq(-2, 2, 0.01), mean=row[1], sd=row[2]),
    label=label,
    distribution='normal'
  ))
  dat <<- rbind(dat, data.frame(
    x=seq(0, 1, 0.01),
    density=sigmoid.dist(seq(0, 1, 0.01), mean=row[1], sd=row[2]),
    label=label,
    distribution='sigmoid'
  ))
}

apply(settings, 1, each.row)

p = ggplot(dat, aes(x))
p = p + geom_line(aes(y = density, colour=distribution))
p = p + facet_grid(label ~ ., labeller=label_parsed)
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave("../report/graphics/theory/batch-norm-activation-distribution.pdf", p, width=12.96703, height=10, units="cm")
