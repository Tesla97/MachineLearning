# Created by Nicola Corea , Software Engineer
# Project For Machine Learning Course
from sklearn.base import clone
from sklearn.pipeline import _name_estimators
import numpy as np
import operator

class HardVotingClassifier(object):

    # estimators  : must be iterable
    # nEstimators : estimators name

    #costruttore
    def __init__(self,estimators):
        self.estimators  = estimators
        self.nEstimators = {key : value for key , value in _name_estimators(estimators)}

    #cAddestrato : fitted classifier
    #fittedC     : fitted classifiers saved

    #fit method
    def fit(self,X_train,y_train):
        # memorizzazione classificatori addestrati
        self.fittedC = []
        # addestramento singoli predittori
        for classificatore in self.estimators:
            cAddestrato = clone(classificatore).fit(X_train,y_train)
            self.fittedC.append(cAddestrato)
        return self
    
    #predict method
    def predict(self,X_test):
        # predizioni singoli classificatori
        predictions = np.asarray([
            classifier.predict(X_test) for classifier in self.fittedC
        ]).T
        # votazione a maggioranza = HARD VOTING
        votazioni   = np.apply_along_axis(lambda x : np.argmax(np.bincount(x)),axis=1,arr=predictions)
        # fine
        return votazioni
    

    

