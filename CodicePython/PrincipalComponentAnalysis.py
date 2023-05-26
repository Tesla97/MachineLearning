# Created by Nicola Corea , Software Engineer
# Project For Machine Learning Course
# Last Modify : 13 / 05 / 2023  12:27
from sklearn.preprocessing   import StandardScaler
from sklearn.preprocessing   import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics         import accuracy_score
from sklearn.decomposition   import PCA
from sklearn.svm             import SVC
import matplotlib.pyplot     as     plt
import pandas as pd
import numpy  as np
import os


if __name__ == '__main__':
    # caricamento dataset red quality wine da UCI repository
    URL         = os.path.join('https://archive.ics.uci.edu','ml','machine-learning-databases','wine-quality','winequality-red.csv') 
    # salvataggio come dataframe pandas
    df_wine     = pd.read_csv(URL,encoding='utf-8',sep=';',quotechar='*')
    # caratteristiche e label classi come array numpy
    X           = df_wine.iloc[:,0:11].values
    y           = df_wine.iloc[:,11].values
    # trasformazione label in low , medium , high quality
    for k in range(len(y)):
        if(y[k] <= 4):
            y[k] = 0   # low quality
        elif(y[k] >= 5 and y[k] <= 7):
            y[k] = 1   # medium quality
        else:
            y[k] = 2   # high quality
    # cross validation
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
    # standardizzazione
    sc          = StandardScaler()
    X_train_std = sc.fit_transform(X_train)
    X_test_std  = sc.transform(X_test)

    # analisi ai componenti principali:
    #
    # 1) Costruzione Matrice Covarianza Caratteristiche 11 x 11
    matriceCov  = np.cov(X_train_std.T)
    # 2) Ricerca Autovalori e Autovettori Matrice Covarianza
    autovalori , autovettori = np.linalg.eig(matriceCov)
    # 3) Calcolo Rapporti Varianza Spiegata 
    sommaAutovalori  = sum(autovalori)
    varianzaSpiegata = [(i / sommaAutovalori) for i in sorted(autovalori,reverse=True)]
    # 4) Somma Cumulativa Varianze Spiegate
    sommaCumulativa  = np.cumsum(varianzaSpiegata)
    # 5) Visualizzazione Varianze Spiegate e Somme Cumulative Per Decidere Quante Componenti Prendere
    plt.figure(1)
    plt.bar(range(1,12),varianzaSpiegata,alpha=0.5,align='center',label='Explained Variance Ratio')
    plt.step(range(1,12),sommaCumulativa,where='mid',label='Cumulative Sums')
    plt.legend(loc='best')
    plt.show()
    # 6) Selezione 5 Autovettori Corrispondenti Ai Primi 5 Autovalori Maggiorni (PCA)
    coppie           = [(np.abs(autovalori[i]),autovettori[:,i]) for i in range(len(autovalori))]
    coppie.sort(key=lambda k: k[0] , reverse=True)
    # 7) Costruzione Matrice Cambio Di Base 
    W                = np.hstack((coppie[0][1][:,np.newaxis],
                                  coppie[1][1][:,np.newaxis],
                                  coppie[2][1][:,np.newaxis],
                                  coppie[3][1][:,np.newaxis],
                                  ))
    # 8) Trasformazione
    X_pca_train      = X_train_std.dot(W)
    X_pca_test       = X_test_std.dot(W)

    # addestramento SVM Kerner , kernel rbf
    svm         = SVC(kernel='rbf',C=10.0,gamma=0.25,random_state=0)
    # addestramento
    svm.fit(X_pca_train,y_train)
    print('Accuratezza (SVM ADD | NO PCA): %.2f'%accuracy_score(y_train,svm.predict(X_pca_train)))
    print('Accuratezza (SVM TEST| NO PCA): %.2f'%accuracy_score(y_test,svm.predict(X_pca_test)))



    

