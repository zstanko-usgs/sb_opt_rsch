scenario1 <- read.csv('scenario1_result.csv')




#scatterplot<- pairs(scenario1_objective0)  this would create a scattermatrix for all values 

#find correlation of all the values
correlation_total<-cor(scenario1)
#make csv file
write.csv(correlationlist_total, file = 'correlation_total.csv')

#make new table for finding correlations above 0.75 (high correlations)
highcorrelations <- cor(scenario1_objective0)
#assign 'NA' to any correlation under 0.75
highcorrelations[abs(highcorrelations) < 0.75] <- NA
#create CSV
write.csv(highcorrelations, file = 'highcorrelations.csv')







