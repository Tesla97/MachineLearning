# Created by Nicola Corea , software engineer
# Project For Machine Learning Course
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from itertools import combinations
import numpy as np

class SequentialBackwardSelection(object):

    # costruttore
    def __init__(self,stimatore,k_f,test_size=0.30,random_state=1):
        self.stimatore    = stimatore
        self.k_f          = k_f
        self.test_size    = test_size
        self.random_state = random_state

    # fit method
    def fit(self,X,y):
        # cross validation
        X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=self.random_state,test_size=self.test_size)
        # numero iniziale caratteristiche dataset
        nC = X.shape[1]
        # memorizzazione indici ottimi (inizialmente tutti)
        self.indici   = tuple(range(nC))
        # memorizzazione sottoinsiemi caratteristiche
        self.sottoIns = [self.indici]
        # validazione modello con tutti gli indici
        self.scores   = [self.valutaIndici(X_train,X_test,y_train,y_test,self.indici)]
        # ottimizzazione indici
        while nC  > self.k_f:
            # memorizzazione score per ogni sottoinsieme di indici
            scores   = []
            sottoIns = []
            # presa in considerazione di tutte le combinazioni di nC - 1 indici
            for p in combinations(self.indici,r = nC - 1):
                # valutazione modello su sottoinsieme p di caratteristiche
                scoreModello = self.valutaIndici(X_train,X_test,y_train,y_test,p)
                # memorizzazione sottoinsieme di indici e score
                scores.append(scoreModello)
                sottoIns.append(p)
            # selezione sottoinsieme ottimale
            indiceBest   = np.argmax(scores)
            self.indici  = sottoIns[indiceBest]
            self.sottoIns.append(self.indici)
            self.scores.append(scores[indiceBest])
            # aggiornamento
            nC           = nC - 1
    
    # transform method
    def transform(self,X):
        return X[:,self.indici]


    # valutazioneIndici method
    def valutaIndici(self,X_train,X_test,y_train,y_test,indici):
        # addestramento modello su indici
        self.stimatore.fit(X_train[:,indici],y_train)
        # ritorno accuratezza
        return accuracy_score(y_test,self.stimatore.predict(X_test[:,indici]))
    
