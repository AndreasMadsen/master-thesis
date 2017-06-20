
library(ggplot2)

relu = function (x) {
  return(ifelse(x > 0, x, 0 * x));
}

selu = function (x) {
  alpha = 1.6732632423;
  lambda = 1.0507009873;
  return(lambda * ifelse(x > 0, x, alpha * (exp(x) - 1)))
}

xlim = seq(-5, 5, 0.01)

dat = data.frame(
  x=xlim,
  value=relu(xlim),
  activation='ReLU'
);

dat = rbind(dat, data.frame(
  x=xlim,
  value=selu(xlim),
  activation='SELU'
));

p = ggplot(dat, aes(x))
p = p + geom_line(aes(y = value, colour=activation))
p = p + theme(legend.position="bottom",
              text=element_text(size=10))

ggsave("../report/graphics/theory/selu-activation-function.pdf", p, width=12.96703, height=7, units="cm")
