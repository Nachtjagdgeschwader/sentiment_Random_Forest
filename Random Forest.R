library(tm)
library(RTextTools)
# Loading train data. Manually coded data stored in the rows 1 to 682.
data <- read.csv2("Data1.csv", header = T, sep = ";")
doc_matrix <- create_matrix(data$Text, language="russian", removeNumbers=TRUE, stemWords=TRUE, removeSparseTerms=.998)
container <- create_container(doc_matrix, data$Sentiment, trainSize=1:682,
                              testSize=683:2378, virgin=FALSE)
RF <- train_model(container,"RF")
# Applying the model (virgin data stored in the rows 1 to 2378):
data <- read.csv2("Data1.csv", header = T, sep = ";")
doc_matrix <- create_matrix(data$Text, language="russian", removeNumbers=TRUE, stemWords=TRUE, removeSparseTerms=.998)
container <- create_container(doc_matrix, data$Sentiment, testSize=683:2378, virgin=T)
RF_CLASSIFY <- classify_model(container, RF)
# Writng the results of classification to a file:
write.table(RF_CLASSIFY, "Predicted.csv", sep = ";")
