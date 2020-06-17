

#sc1 <- read.csv ('scenario1_result.csv')
sc1_pca_loadings <- prcomp(sc1)
sc1_pca_loadings
#write.csv(sc1_pca_loadings, 'sca_pca_loadings.csv')

summary(sc1_pca_loadings)



#----
S <- cov(sc1)
S

sum(diag(S))

s.eigen <- eigen(S)
s.eigen

for (s in s.eigen$values) {
  print(s / sum(s.eigen$values))
}

sc1_loadings<- s.eigen$vectors
write.csv(sc1_loadings, 'sc1_loadings.csv')

#pc1 vs pc2 plot
install.packages("ggfortify")
library(ggfortify)
autoplot(sc1_pca_loadings)
