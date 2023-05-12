# Created by Nicola Corea , Software Engineer 
# Project for the course of Machine Learning
# 11/05/2023
import pandas as pd
import numpy  as np
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# definizione ensemble
from HardVotingClassifier import HardVotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier



if __name__ == '__main__':
    # loading dataset from machine learning repositories
    URL         = os.path.join('https://archive.ics.uci.edu','ml','machine-learning-databases','wine-quality','winequality-red.csv') 
    # saving as Pandas Dataframe
    df_wine     = pd.read_csv(URL,encoding='utf-8',sep=';',quotechar='*')
    # separing the samples from target values as numpy array
    X           = df_wine.iloc[:,0:11].values
    y           = df_wine.iloc[:,11].values
    # label trasformation , starting from 0
    le          = LabelEncoder()
    y           = le.fit_transform(y)
    # cross validation method for testing model
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
    # standardization
    sc          = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std  = sc.transform(X_test)
    # classifier training
    svm         = SVC(kernel='rbf',C=10.0,gamma=0.25,random_state=0,probability=True)
    tree        = DecisionTreeClassifier(max_depth=5,criterion='entropy',random_state=0)
    knn         = KNeighborsClassifier(n_neighbors=10,p=2,metric='minkowski')
    # training
    svm.fit(X_train_std,y_train)
    tree.fit(X_train_std,y_train)
    knn.fit(X_train_std,y_train)
    # validation
    print('\n')
    print('Accuracy (SVM | TRAIN)  : %.2f'%accuracy_score(y_train,svm.predict(X_train_std)))
    print('Accuracy (SVM | TEST)   : %.2f\n'%accuracy_score(y_test,svm.predict(X_test_std)))
    print('Accuracy (DT  | TRAIN)  : %.2f'%accuracy_score(y_train,tree.predict(X_train_std)))
    print('Accuracy (DT  | TEST)   : %.2f\n'%accuracy_score(y_test,tree.predict(X_test_std)))
    print('Accuracy (KNN | TRAIN)  : %.2f'%accuracy_score(y_train,knn.predict(X_train_std)))
    print('Accuracy (KNN | TEST)   : %.2f\n'%accuracy_score(y_test,knn.predict(X_test_std)))
    # ensemble classificator
    majorityVoting  = HardVotingClassifier(estimators=[svm,tree,knn])
    baggingEnsemble = BaggingClassifier(n_estimators=25,bootstrap=True,bootstrap_features=True,random_state=0)
    boostingEnsemble= AdaBoostClassifier(n_estimators=25,learning_rate=0.1,random_state=0) 
    majorityVoting.fit(X_train_std,y_train)
    baggingEnsemble.fit(X_train_std,y_train)
    boostingEnsemble.fit(X_train_std,y_train)
    print('Accuracy (HV | TRAIN)  : %.2f'%accuracy_score(y_train,majorityVoting.predict(X_train_std)))
    print('Accuracy (HV | TEST)   : %.2f\n'%accuracy_score(y_test,majorityVoting.predict(X_test_std)))
    print('Accuracy (BA | TRAIN)  : %.2f'%accuracy_score(y_train,baggingEnsemble.predict(X_train_std)))
    print('Accuracy (BA | TEST)   : %.2f\n'%accuracy_score(y_test,baggingEnsemble.predict(X_test_std)))
    print('Accuracy (AB | TRAIN)  : %.2f'%accuracy_score(y_train,boostingEnsemble.predict(X_train_std)))
    print('Accuracy (AB | TEST)   : %.2f\n'%accuracy_score(y_test,boostingEnsemble.predict(X_test_std)))



