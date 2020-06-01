#read in scenario1 data
scenario1<- read.csv('scenario1_result.csv')
#11 wells per quarter, 4 quarters in a year -> 44 columns per year

yr1working <- scenario1[,1:44] 
yr1 <- stack(yr1working)

yr2working <- scenario1[,45:88] 
yr2 <- stack(yr2working)

yr3working <- scenario1[,89:132] 
yr3 <- stack(yr3working)

yr4working <- scenario1[,133:176] 
yr4 <- stack(yr4working)

yr5working <- scenario1[,177:220] 
yr5 <- stack(yr5working)

yr6working <- scenario1[,221:264] 
yr6 <- stack(yr7working)

yr7working <- scenario1[,265:308] 
yr7 <- stack(yr7working)

yr8working <- scenario1[,309:352] 
yr8 <- stack(yr8working)

yr9working <- scenario1[,353:396] 
yr9 <- stack(yr9working)

yr10working <- scenario1[,397:440] 
yr10 <- stack(yr10working)


boxplot(yr1$values, yr2$values,yr3$values,yr4$values,yr5$values,yr6$values,yr7$values,yr8$values,yr9$values,yr10$values, main = 'Pump Rates by Year',
        names = c('1','2','3','4','5','6','7','8','9','10'), xlab = "Year", ylab = 'Pump Rate')

par(mfrow=c(2, 5))
hist(yr1$values, main = 'Year 1', xlab = 'Pump Rates')
hist(yr2$values,main = 'Year 2', xlab = 'Pump Rates')
hist(yr3$value,main = 'Year 3', xlab = 'Pump Rates')
hist(yr4$values,main = 'Year 4', xlab = 'Pump Rates')
hist(yr5$values,main = 'Year 5', xlab = 'Pump Rates')
hist(yr6$values,main = 'Year 6', xlab = 'Pump Rates')
hist(yr7$values,main = 'Year 7', xlab = 'Pump Rates')
hist(yr8$values,main = 'Year 8', xlab = 'Pump Rates')
hist(yr9$values,main = 'Year 9', xlab = 'Pump Rates')
hist(yr10$values,main = 'Year 10', xlab = 'Pump Rates')


