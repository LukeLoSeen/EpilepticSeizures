############################################IMPORTANT#############################################
#Le code commenté ci-dessous ne fournit qu'une partie de l'information produite. L'explication des
#techniques utilisées, ainsi que la présentation et l'interprétation des résultats sont présentées dans
#un rapport annexe.
##################################################################################################


setwd("C:/Users/YourName/epileptic_seizures")

#install.packages("pROC")
#install.packages("FactoMineR")
#install.packages("Factoshiny")
#install.packages("randomForest")
#install.packages("class")
#install.packages("doParallel")

#On importe l'échantillon
seizures<-read.csv("data_seizures.csv")
seizures<-seizures[,-1]
#La première colonne correspond à l'identifiant de l'observation: on la supprime car celle-ci est inutile.
#pire encore: elle pourrait causer des problèmes d'overfitting.

colnames(seizures)[179]<-"seizure"

library(FactoMineR)
library(Factoshiny)
library(randomForest)
library(class)
library(doParallel)
library(ggplot2)
library(dplyr)
library(pROC)

detectCores()
getDoParWorkers()
registerDoParallel(cores = 3)
#Nous augmentons le nombre de coeurs alloués à R afin d'augmenter la rapidité d'exécution du code.
#Il est conseillé d'allouer n-1 coeurs à R avec n le nombre total de coeurs de l'ordinateur (égal à 4 dans notre cas)

n=nrow(seizures)

res<-seizures[,179]
for (i in 1:n)
{
  if(seizures[i,179]==1)
  {
    res[i]<-1
  }
  else
  {
    res[i]<-0
  }
}

seizures[,179]<-res

#Les 178 premières colonnes sont un enregistrement de l'activité cérébrale pendant une seconde (projeté dans 
#un espace de dimension 178)tirée aléatoirement d'un des 500 patients (cf. rapport) tiré aléatoirement aussi.

#La variable "seizure" contient une valeur de 1 à 5 décrivant l'état du patient durant l'encéphalographie.
#1) si la variable vaut 5, le patient avait les yeux ouverts pendant l'enregisterment
#2) si elle vaut 4, le patient avait les yeux fermés lors de l'enregistrement
#3) si elle vaut 3, il s'agit d'un patient pour lequel on a identifié la zone du cerbeau où se trouve la tumeur
#et il s'agit de l'enregistrement de l'activité cérébrale de la partie saine du cerveau
#4) si elle vaut 2, il s'agit d'un patient pour lequel on a identifié la zone du cerbeau où se trouve la tumeur
#et il s'agit de l'enregistrement de la partie tuméreuse du cerveau
#5) si ekke vaut 1, une crise d'épilepsie a eu lieu pendant l'enregistrement.

#Par souci de simplicité, nous avons remplacé cette variable par une variable binaire qui renvoie 1 si
#il s'agit d'une crise d'épilepsie, et 0 sinon.

table(seizures[,179])
#l'échantillon contient 2300 crises d'épilepsies pour 9200 observations sans crise, soit 20% de positifs.
#Il s'agit d'une proportion acceptable.


#On divise l'échantillon de base en un échantillon d'entraînement et un échantillon test:

N<-floor(n*0.7) #proportion d'observations dans l'échantillon d'entraînement. Ici, 0.7
sample1<-sample(1:11500,N,replace=T)
train_sample<-seizures[sample1,]
test_sample<-seizures[-sample1,]

#Entraînement du modèle des k plus proches voisins

res<-knn(train_sample[,-179],test_sample[,-179],train_sample[,179])
res<-as.numeric(as.vector(res))

#prédiction sur l'échantillon test

prediction <- knn(train_sample[,-179], test_sample[,-179], train_sample[,179], k = 1)

m.confusiontest <- table(prediction, test_sample[,179])
m.confusiontest

txprecision <- (m.confusiontest[4]/(m.confusiontest[4]+m.confusiontest[3]))
txprecision

txrappel <- (m.confusiontest[4]/(m.confusiontest[4]+m.confusiontest[2]))
txrappel

txbiensclassées <- ((m.confusiontest[4]+m.confusiontest[1])/(m.confusiontest[4]+m.confusiontest[2]+m.confusiontest[1]+m.confusiontest[3]))
txbiensclassées

#Taux d'observations bien classées en fonction de k

accuracy <- rep(0,10)
k <- 1:10
for (x in k) { 
  prediction <- knn(train_sample[,-179], test_sample[,-179], train_sample[,179], k = x)
  m.confusiontest <- table(prediction, test_sample[,179])
  accuracy[x] <- ((m.confusiontest[4]+m.confusiontest[1])/(m.confusiontest[4]+m.confusiontest[2]+m.confusiontest[1]+m.confusiontest[3]))
}

plot(k, accuracy, type = 'b')

plot <- qplot(y = accuracy, x = k, geom = "line")
plot 

#entraînement du Random Forest

train_sample$seizure<-as.factor(train_sample$seizure)
forest<-randomForest(seizure~.,data=train_sample,na.action = na.omit)

#erreur OOB en fonction du nombre d'arbres

plot(forest$err.rate[,1],xlab="number of trees",ylab="OOB error",type="l")

#courbe ROC
result.predicted.prob <- predict(forest, test_sample, type="prob")

test_sample$predictions<-as.numeric(test_sample$predictions)
rc<-roc(test_sample$seizure,result.predicted.prob[,2])
plot.roc(rc,print.thres="best", print.thres.best.method="closest.topleft")

#Matrice de confusion

predictions<-ifelse(result.predicted.prob[,2]<0.361,0,1)
table(predictions,test_sample$seizure)




