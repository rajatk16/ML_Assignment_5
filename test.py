#!/usr/bin/env python3

# Import Libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

# Import the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
# print(dataset.shape)
# print(dataset.columns.values)


# Divide the data into X and Y
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

X1 = dataset.iloc[:, 3:13]
# print(X1.head(5))

# Using OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[1])
ct = ColumnTransformer(
  [
    (
      'one_hot_encoder', 
      OneHotEncoder(), 
      [1,2]
    )
  ],
  remainder="passthrough"
)
X = np.array(ct.fit_transform(X))
X = X[:,1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer='glorot_uniform', activation='relu'))
classifier.add(Dense(units=6, kernel_initializer='glorot_uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train, y_train, batch_size=10, epochs=20)

y_pred = classifier.predict(X_test)

# cm = confusion_matrix(y_test, y_pred)

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

def build_classifier():
  classifier = Sequential()
  classifier.add(Dense(units=6, kernel_initializer='glorot_uniform', activation='relu'))
  classifier.add(Dense(units=6, kernel_initializer='glorot_uniform', activation='relu'))
  classifier.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
  classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
  return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, nb_epoch=3)
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10, n_jobs=1)

print(accuracies.mean())
print(accuracies.std())

from keras.layers import Dropout

classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer="glorot_uniform", activation='relu'))
classifier.add(Dropout(rate=0.1))
classifier.add(Dense(units=6, kernel_initializer="glorot_uniform", activation='relu'))
classifier.add(Dropout(rate=0.1))

classifier.add(Dense(units=1, kernel_initializer="glorot_uniform", activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

def build_classifier(optimizer):
  classifier = Sequential()
  classifier.add(Dense(units=6, kernel_initializer="glorot_uniform", activation="relu"))
  classifier.add(Dense(units=6, kernel_initializer="glorot_uniform", activation="relu"))
  classifier.add(Dense(units=1, kernel_initializer="glorot_uniform", activation="sigmoid"))
  classifier.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
  return classifier

classifier = KerasClassifier(build_fn=build_classifier)
parameters = {
  'batch_size': [25,32], 
  'nb_epoch':[1,2], 
  'optimizer': ['adam', 'rmsprop']
}
grid_search = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=10)
grid_search = grid_search.fit(X_train, y_train)

best_parameters = grid_search.best_params_
best_accuracy = grid_search.best_score_

print(best_parameters)
print(best_accuracy)