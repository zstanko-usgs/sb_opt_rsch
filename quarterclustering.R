#````
#912 rows *11wells *10 years= number of rows from all years for 1 quarter = 100320
#Each dataset should have 10032 rows. Multiply by 10 columns (1 per year) to = 100320
nrow(split_q1)
nrow(split_q2)
nrow(split_q3)
nrow(split_q4)
#```

#read in scenario1 data
scenario1<- read.csv('scenario1_result.csv')

#Separate quarters from each other. Stack columns.
#quarter1
q1sorter <- numeric()
for (i in seq(1, 440, 44)) {
  q1sorter <- c(q1sorter, i + 0:10)
}
quarter1 <- stack(scenario1[,q1sorter])
#quarter2
q2sorter <- numeric()
for (i in seq(12, 440, 44)) {
  q2sorter <- c(q2sorter, i + 0:10)
}
quarter2 <- stack(scenario1[,q2sorter])
#quarter3
q3sorter <- numeric()
for (i in seq(23, 440, 44)) {
  q3sorter <- c(q3sorter, i + 0:10)
}
quarter3 <- stack(scenario1[,q3sorter])
#quarter4
q4sorter <- numeric()
for (i in seq(34, 440, 44)) {
  q4sorter <- c(q4sorter, i + 0:10)
}
quarter4 <- stack(scenario1[,q4sorter])


#Unstack columns and seperate into equal columns (each new column represents a new year)
#Quarter1
split_q1<-matrix(quarter1[,1], nrow=10032, ncol=10)
#Quarter2
split_q2<-matrix(quarter2[,1], nrow=10032, ncol=10)
#Quarter3
split_q3<-matrix(quarter3[,1], nrow=10032, ncol=10)
#Quarter4
split_q4<-matrix(quarter4[,1], nrow=10032, ncol=10)
#Add column names (number corresponds to year in the study)
colnames(split_q1) <- c("1","2","3","4","5","6","7", "8", "9","10")
colnames(split_q2) <- c("1","2","3","4","5","6","7", "8", "9","10")
colnames(split_q3) <- c("1","2","3","4","5","6","7", "8", "9","10")
colnames(split_q4) <- c("1","2","3","4","5","6","7", "8", "9","10")


#Analysis

#Quarter 1 Analysis
#histograms
par(mfrow=c(2, 5))
hist(split_q1[,1], main = 'Year 1', xlab = 'Pump Rate')
hist(split_q1[,2], main = 'Year 2', xlab = 'Pump Rate')
hist(split_q1[,3], main = 'Year 3', xlab = 'Pump Rate')
hist(split_q1[,4], main = 'Year 4', xlab = 'Pump Rate')
hist(split_q1[,5], main = 'Year 5', xlab = 'Pump Rate')
hist(split_q1[,6], main = 'Year 6', xlab = 'Pump Rate')
hist(split_q1[,7], main = 'Year 7', xlab = 'Pump Rate')
hist(split_q1[,8], main = 'Year 8', xlab = 'Pump Rate')
hist(split_q1[,9], main = 'Year 9', xlab = 'Pump Rate')
hist(split_q1[,10], main = 'Year 10', xlab = 'Pump Rate')

#Quarter 2 Analysis
#histograms
par(mfrow=c(2, 5))
hist(split_q2[,1], main = 'Year 1', xlab = 'Pump Rate')
hist(split_q2[,2], main = 'Year 2', xlab = 'Pump Rate')
hist(split_q2[,3], main = 'Year 3', xlab = 'Pump Rate')
hist(split_q2[,4], main = 'Year 4', xlab = 'Pump Rate')
hist(split_q2[,5], main = 'Year 5', xlab = 'Pump Rate')
hist(split_q2[,6], main = 'Year 6', xlab = 'Pump Rate')
hist(split_q2[,7], main = 'Year 7', xlab = 'Pump Rate')
hist(split_q2[,8], main = 'Year 8', xlab = 'Pump Rate')
hist(split_q2[,9], main = 'Year 9', xlab = 'Pump Rate')
hist(split_q2[,10], main = 'Year 10', xlab = 'Pump Rate')

#Quarter 3 Analysis
#histograms
par(mfrow=c(2, 5))
hist(split_q3[,1], main = 'Year 1', xlab = 'Pump Rate')
hist(split_q3[,2], main = 'Year 2', xlab = 'Pump Rate')
hist(split_q3[,3], main = 'Year 3', xlab = 'Pump Rate')
hist(split_q3[,4], main = 'Year 4', xlab = 'Pump Rate')
hist(split_q3[,5], main = 'Year 5', xlab = 'Pump Rate')
hist(split_q3[,6], main = 'Year 6', xlab = 'Pump Rate')
hist(split_q3[,7], main = 'Year 7', xlab = 'Pump Rate')
hist(split_q3[,8], main = 'Year 8', xlab = 'Pump Rate')
hist(split_q3[,9], main = 'Year 9', xlab = 'Pump Rate')
hist(split_q3[,10], main = 'Year 10', xlab = 'Pump Rate')

#Quarter 3 Analysis
#histograms
par(mfrow=c(2, 5))
hist(split_q4[,1], main = 'Year 1', xlab = 'Pump Rate')
hist(split_q4[,2], main = 'Year 2', xlab = 'Pump Rate')
hist(split_q4[,3], main = 'Year 3', xlab = 'Pump Rate')
hist(split_q4[,4], main = 'Year 4', xlab = 'Pump Rate')
hist(split_q4[,5], main = 'Year 5', xlab = 'Pump Rate')
hist(split_q4[,6], main = 'Year 6', xlab = 'Pump Rate')
hist(split_q4[,7], main = 'Year 7', xlab = 'Pump Rate')
hist(split_q4[,8], main = 'Year 8', xlab = 'Pump Rate')
hist(split_q4[,9], main = 'Year 9', xlab = 'Pump Rate')
hist(split_q4[,10], main = 'Year 10', xlab = 'Pump Rate')


#Comparing quarterly boxplots to each other
par(mfrow=c(2, 2))
boxplot(split_q1[,1:10], xlab = "Year", ylab = 'Pump Rate', main = 'Quarter 1 by Year')
boxplot(split_q2[,1:10], xlab = "Year", ylab = 'Pump Rate', main = 'Quarter 2 by Year')
boxplot(split_q3[,1:10], xlab = "Year", ylab = 'Pump Rate', main = 'Quarter 3 by Year')
boxplot(split_q4[,1:10], xlab = "Year", ylab = 'Pump Rate', main = 'Quarter 4 by Year')







