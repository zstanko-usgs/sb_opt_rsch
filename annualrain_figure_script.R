library(ggplot2)
library(dplyr)
library(plyr)

rain_yr <- read.csv('rain_mean_annual.csv')
#Time Series
plot <- ggplot(rain_yr, aes(x=year, y=mean_precip)) +
  geom_line() + 
  geom_line(color="royalblue1")+
  xlab("Year") +
  ylab("Precipitation (inches)") +
  scale_x_continuous(breaks = seq(1972, 2018, by = 5)) +
  ggtitle("Mean Annual Precipitation in Santa Barbara County", 
          subtitle = "1972 to 2018") 
plot




rain_qtr <- read.csv('rain_mean_quarter_transposed.csv', check.names = FALSE)
par(mar=c(4,4,4,4))
boxplot(rain_qtr, xlab = 'Quarter', ylab = 'Precipitation (inches)',
        main = 'Mean Quarterly Precipitation in SB County \n (1972 to 2018)')
