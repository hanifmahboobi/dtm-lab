#! /path/to/Rscript

library(stats)
library(strucchange)

topic_times <- read.table("./models/db/topic_times.csv", header=F, sep=",")
time_start <- read.table("./models/db/time-seq.txt", header=F)
col <- length(topic_times[1,]) - 1
topic_num <- c(1:col)
print(topic_num)

for(i in topic_num){

  png(file=paste("./Figures/strucchange_topic",as.character(i-1),".png"), width=1436, height=677)

  prob <- ts(topic_times[,i], start=c(as.numeric(time_start[1,1])), frequency=1)

  # store the breakdates
  bp_ts <- breakpoints(prob~1)

  # this will give you the break dates and their confidence intervals
  summary(bp_ts)

  # to plot the breakpoints with confidence intervals
  plot(prob, xlab="Year", ylab="Probability",main=paste("Topic",as.character(i-1)))
  lines(bp_ts)
  dev.off()
}