library(tidyverse)
library(data.table)
#find annual mean
rain <- read.csv('source_data/rain.csv')
rain_mean_annual<-aggregate( daily.rain ~ year, rain, mean )
rain_mean_annual<-rain_mean_annual%>% 
  rename(
    mean_precip = daily.rain
  )
#write.csv(rain_mean_annual,'rain_mean_annual.csv')

#find quarterly mean
rain_quarter <- read.csv('source_data/rain_quarters.csv')
rain_mean_quarter<- dcast(setDT(rain_quarter), quarter ~ year, value.var = 'daily.rain', mean)
#write.csv(rain_mean_quarter,'rain_mean_quarter.csv')

