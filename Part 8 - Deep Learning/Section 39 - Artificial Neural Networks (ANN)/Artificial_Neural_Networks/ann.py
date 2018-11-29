#importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#data preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2= LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)



from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Making the artificial neural network

import keras 
from keras.models import Sequential
from keras.layers import Dense

#initializing the neural network
classifier = Sequential()

#adding the input layer , hidden layers and output layers 
classifier.add(Dense(output_dim =6,init = 'uniform',activation = 'relu',input_dim = 11))
classifier.add(Dense(ouput_dim = 6,init = 'uniform',activation = 'relu'))
classifier.add(Dense(output_dim = 1,init = 'uniform',activation = 'sigmoid'))

#Applying stochastic gradient descent 

classifier.compile(optimizer = 'adam',loss ='binary_crossentropy',metrics=['accuracy'])

#Fitting the ANN
classifier.fit(X_train,y_train,batch_size=10,nb_epoch = 100)

#making the predictions and evaluating the model
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)


