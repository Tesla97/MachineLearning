# Created by Nicola Corea , Software Engineer
# Project For The Course Of Machine Learning
# Last Modified: 15 / 05 / 2023
from   sklearn.model_selection import train_test_split
from   sklearn.preprocessing   import StandardScaler
import matplotlib.pyplot       as plt
import tensorflow              as tf
import pandas as pd
import numpy  as np
import os

if __name__ == '__main__':
  # caricamento dataset red quality wine dalla rete
  URL         = os.path.join('https://archive.ics.uci.edu','ml','machine-learning-databases','wine-quality','winequality-red.csv') 
  # caricamento dataset come un Dataframe Pandas
  df_wine     = pd.read_csv(URL,encoding='utf-8',sep=';',quotechar='*')
  # separazione caratteristiche dalle classi
  X           = df_wine.iloc[:,0:11].values
  y           = df_wine.iloc[:,11].values
  # label transformation in low, medium , high = (0,1,2)
  for k in range(len(y)):
      if(y[k] <= 4):
          y[k] = 0   # low quality
      elif(y[k] >= 5 and y[k] <= 7):
          y[k] = 1   # medium quality
      else:
          y[k] = 2   # high quality
  # cross validation
  X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1,stratify=y)
  # standardizzation
  sc          = StandardScaler()
  X_train_std = sc.fit_transform(X_train)
  X_test_std  = sc.transform(X_test)
  # addestramento di un perceptron multi-layer
  mlp         = tf.keras.Sequential([
      tf.keras.layers.Dense(16,activation='relu',name='fc1',input_shape=(11,)),
      tf.keras.layers.Dense(3,name='fc3',activation='softmax')
  ])
  # compilazione
  mlp.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
  # addestramento
  history = mlp.fit(X_train_std,y_train,epochs=20,verbose=1)
  hist    = history.history
  # Evaluate the model on test data
  plt.figure(1)
  plt.plot(hist['loss'],lw=2)
  plt.title('Training Loss')
  plt.figure(2)
  plt.plot(hist['accuracy'],lw=2)
  plt.title('Training Accuracy')
  # valutazione su dati test
  results = mlp.evaluate(X_test_std,y_test,verbose=0)
  print('Test loss: {:.4f}  | Test Accuracy: {:.4f}'.format(*results))
