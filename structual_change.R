#!/path/to/Rscript

library(stats)
library(strucchange)

topic_times <- read.table("./db/topic_times.csv", header=F, sep=",")
topic_num <- c(1:length(topic_times[1,]))
print(topic_num)

for(i in topic_num){

  png(file=paste("./Figures/strucchange_topic_",as.character(i-1),".png"))

  prob <- ts(topic_times[,i], start=c(1948), frequency=1)

  # store the breakdates
  bp_ts <- breakpoints(prob~1)

  # this will give you the break dates and their confidence intervals
  summary(bp_ts) 

  # store the confidence intervals
  ci_ts <- confint(bp_ts, TRUE)

  # to plot the breakpoints with confidence intervals
  plot(prob, xlab="Year", ylab="Probability",main=paste("Topic",as.character(i-1)))
  lines(bp_ts)
  lines(ci_ts)
  dev.off()
}