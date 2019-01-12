# ---------------- Install packages
library(matrixStats)
library(ggplot2)
library(factoextra)
library(dplyr)
library(stats)
library(fpc)
library(arules)
install.packages("arulesCBA")
library(arulesCBA)
library(arulesViz)
library(mclust)
install.packages("party", dependencies = TRUE)
library(party)
library(tree)

# ---------------- Read and inspect data
bank <- read.table("C:/Users/Amelia/Desktop/MACHINE LEARNING/data_banknote_authentication.txt",sep = ",")
colnames(bank) <- c("variance", "skewness", "curtosis", "entropy", "class")
head(bank)
str(bank)
class(bank)
dim(bank)
summary(bank)
View(bank)

# --------------- Presenting the data
hist(bank[,1],
     main = "Histogram for Variance", xlab = "Variance",
     breaks = 10, col="lightgrey")
hist(bank[,2],
     main = "Histogram for Skewness", xlab = "Skewness",
     breaks = 10, col="lightgrey")
hist(bank[,3],
     main = "Histogram for Curtosis", xlab = "Curtosis",
     breaks = 10, col="lightgrey")
hist(bank[,4],
     main = "Histogram for Entropy", xlab = "Entropy",
     breaks = 10, col="lightgrey")
hist(bank[,5],
     main = "Histogram for Class", xlab = "Class",
     breaks = 2, col="lightgrey")

boxplot(bank[,1:5])
title("Coparing boxlots of five attributes")

#---------------- Data Preperation
bc <- bank$class <- as.factor(bank$class)
summary(bank$class)
mr  <- as.matrix(scale(bank[,1:4]))
bank_matrix <- cbind(bc,mr)

unique(bank$class)
x11()
plot(
  x=bank[1:4],
  col= as.integer(bank$class)+2, main= "Scaterplott of each predictors")

bank.cor <- bank_matrix[,2:5]
cor(bank.cor)

pairs(~variance+skewness+curtosis+entropy, data = bank_matrix,
      main="Simple Scatterplot Matrix of Predictors")

# -------------- Mclust function 

bankMclust1 <- Mclust(bank[,1:4], G=2) #run mclust with 2 segments
summary(bankMclust1, parameters=TRUE)
plot(bankMclust1)

bankMclust2 <- Mclust(bank[,1:4], G=3) #run mclust with 3 segments
summary(bankMclust2, parameters=TRUE)
plot(bankMclust2)

bankMclust3 <- Mclust(bank[,1:4]) #mclust decides on number of segments 
summary(bankMclust3, parameters=TRUE)

# --------------- Cluster Analysis

X11()
fviz_nbclust(mr, kmeans, method = "wss")

clusters2 <- kmeans(
  x = mr,
  centers = 2, 
  nstart = 15)
clusters2$cluster
sort(clusters2$cluster)

clusters3 <- kmeans(
  x = mr,
  centers = 3, 
  nstart = 15)
clusters3$cluster

clusters4 <- kmeans(
  x = mr,
  centers = 4, 
  nstart = 15)
clusters4$cluster

# ------------ Graphs for cluster analysis
x11()
fviz_cluster(clusters2,
             data = mr, 
             geom = c("point"),
             ellipse.type = "convex",
             palette="Set2",
             ggtheme = theme_minimal())

x11()
fviz_cluster(clusters3,
             data = mr, 
             geom = c("point"),
             ellipse.type = "convex",
             palette="Set2",
             ggtheme = theme_minimal())

x11()
fviz_cluster(clusters4,
             data = mr, 
             geom = c("point"),
             ellipse.type = "convex",
             palette="Set2",
             ggtheme = theme_minimal())

bank2 <- bank
bank2$class <- NULL
table(bank$class, clusters2$cluster)
table(bank$class, clusters3$cluster)
table(bank$class, clusters4$cluster)

plot(bank2[c("variance", "skewness")],
     col=clusters2$cluster)

plot(bank2[c("variance", "skewness")],
     col=clusters3$cluster)

plot(bank2[c("variance", "skewness")],
     col=clusters4$cluster)

# --------- Hierarchical tree

# first approach
x11()
d1 <- dist(bank_matrix, method = "euclidean") 
fit1 <- hclust(d1, method = "ward.D")
plot(fit1, main = "Hierarchical clustering")

x11()
d2 <- dist(bank_matrix, method = "maximum") 
fit2 <- hclust(d2, method = "ward.D")
plot(fit2, main = "Hierarchical clustering")

x11()
d3 <- dist(bank_matrix, method = "manhattan") 
fit3 <- hclust(d3, method = "ward.D")
plot(fit3, main = "Hierarchical clustering")

# second approach
samp <- sample(1:dim(bank)[1], 50)
bankSample <- bank[samp,]
bankSample$class <- NULL
x11()
hc <- hclust(dist(bankSample), method = "ward.D")
plot(hc, hang= -1, labels = bank$class[idx])

rect.hclust(hc, k=2, border = "blue")


# ------------- PCA
pca <- prcomp(mr)
pca
summary(pca)
screeplot(pca, main ="Scree Plot", col = "lightblue", type = "line")
biplot(pca)


# ----------- Association rules
rules <- apriori(bank)
inspect(rules)

rules1 <- apriori(bank, parameter = list(minlen=4, supp=0.08, conf=0.4),
                  control = list(verbose=F))
rules.sorted <- sort(rules1, by="lift")
inspect(rules.sorted)

#determination the support level
newcolumn <- 1:nrow(bank)
newbank <- cbind(newcolumn, bank)
head(newbank)

supp <- lapply(split(x=newbank[,"class"], 
               f= newbank$newcolumn), unique)
supp <- as(supp, "transactions")
summary(supp)

itemFrequency(supp)
summary(itemFrequency(supp))

#second approach - accessing the rules of the classifier 

classifier <- CBA(class~.,bank, support = 0.05, confidence = 0.9)
classifier

rules <- rules(classifier)
inspect(rules)
rules.sorted <- sort(rules, by="lift")
inspect(rules.sorted)

plot(rules)
plot(rules, method="graph", control = list(type="items"))
plot(rules, method="paracoord", control=list(reorder=TRUE))

# using the classifier 

pred <- predict(classifier, bank)

head(pred)
table(pred)

#checking the results
table(pred, truth = bank$class)


# -------------- DECISIONAL TREES 

set.seed(12345)
bankct <- ctree(class ~ ., data = bank)
bankct

plot(bankct)
table(predict(bankct), bank$class)

tr <- treeresponse(bankct, newbank = bank[1:10,])

banktree <- tree(class ~ variance+skewness+curtosis+entropy,
                 data = bank)
summary(banktree)

plot(banktree)
text(banktree)
title("Decision tree of dataset")
