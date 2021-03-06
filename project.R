############################################IMPORTANT#############################################
#Le code comment� ci-dessous ne fournit qu'une partie de l'information produite. L'explication des
#techniques utilis�es, ainsi que la pr�sentation et l'interpr�tation des r�sultats sont pr�sent�es dans
#un rapport annexe.
##################################################################################################


setwd("C:/Users/YourName/epileptic_seizures")

#install.packages("pROC")
#install.packages("FactoMineR")
#install.packages("Factoshiny")
#install.packages("randomForest")
#install.packages("class")
#install.packages("doParallel")

#On importe l'�chantillon
seizures<-read.csv("data_seizures.csv")
seizures<-seizures[,-1]
#La premi�re colonne correspond � l'identifiant de l'observation: on la supprime car celle-ci est inutile.
#pire encore: elle pourrait causer des probl�mes d'overfitting.

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
#Nous augmentons le nombre de coeurs allou�s � R afin d'augmenter la rapidit� d'ex�cution du code.
#Il est conseill� d'allouer n-1 coeurs � R avec n le nombre total de coeurs de l'ordinateur (�gal � 4 dans notre cas)

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

#Les 178 premi�res colonnes sont un enregistrement de l'activit� c�r�brale pendant une seconde (projet� dans 
#un espace de dimension 178)tir�e al�atoirement d'un des 500 patients (cf. rapport) tir� al�atoirement aussi.

#La variable "seizure" contient une valeur de 1 � 5 d�crivant l'�tat du patient durant l'enc�phalographie.
#1) si la variable vaut 5, le patient avait les yeux ouverts pendant l'enregisterment
#2) si elle vaut 4, le patient avait les yeux ferm�s lors de l'enregistrement
#3) si elle vaut 3, il s'agit d'un patient pour lequel on a identifi� la zone du cerbeau o� se trouve la tumeur
#et il s'agit de l'enregistrement de l'activit� c�r�brale de la partie saine du cerveau
#4) si elle vaut 2, il s'agit d'un patient pour lequel on a identifi� la zone du cerbeau o� se trouve la tumeur
#et il s'agit de l'enregistrement de la partie tum�reuse du cerveau
#5) si ekke vaut 1, une crise d'�pilepsie a eu lieu pendant l'enregistrement.

#Par souci de simplicit�, nous avons remplac� cette variable par une variable binaire qui renvoie 1 si
#il s'agit d'une crise d'�pilepsie, et 0 sinon.

table(seizures[,179])
#l'�chantillon contient 2300 crises d'�pilepsies pour 9200 observations sans crise, soit 20% de positifs.
#Il s'agit d'une proportion acceptable.


#On divise l'�chantillon de base en un �chantillon d'entra�nement et un �chantillon test:

N<-floor(n*0.7) #proportion d'observations dans l'�chantillon d'entra�nement. Ici, 0.7
sample1<-sample(1:11500,N,replace=T)
train_sample<-seizures[sample1,]
test_sample<-seizures[-sample1,]

#Entra�nement du mod�le des k plus proches voisins

res<-knn(train_sample[,-179],test_sample[,-179],train_sample[,179])
res<-as.numeric(as.vector(res))

#pr�diction sur l'�chantillon test

prediction <- knn(train_sample[,-179], test_sample[,-179], train_sample[,179], k = 1)

m.confusiontest <- table(prediction, test_sample[,179])
m.confusiontest

txprecision <- (m.confusiontest[4]/(m.confusiontest[4]+m.confusiontest[3]))
txprecision

txrappel <- (m.confusiontest[4]/(m.confusiontest[4]+m.confusiontest[2]))
txrappel

txbiensclass�es <- ((m.confusiontest[4]+m.confusiontest[1])/(m.confusiontest[4]+m.confusiontest[2]+m.confusiontest[1]+m.confusiontest[3]))
txbiensclass�es

#Taux d'observations bien class�es en fonction de k

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

#entra�nement du Random Forest

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




