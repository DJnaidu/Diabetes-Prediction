import numpy as np
import tensorflow as tf
from numpy import loadtxt
data = loadtxt("/content/diabetes.csv", delimiter=',',skiprows=1)
data.shape
input = data[ : ,0:8]
output = data[ : ,8]
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(input, output, test_size= 0.4)
from keras.layers import Dense
from keras.models import Sequential
model = Sequential()
#Creating neural networks
model.add(Dense(8,input_dim = 8,activation = "relu" )) #input layer
model.add(Dense(8,activation = "relu")) #hidden layer
model.add(Dense(1,activation = "sigmoid")) #output layer
model.compile(loss="binary_crossentropy",optimizer = "adam", metrics = ["accuracy"])
model.fit(input,output,epochs=10)
accuracy = model.evaluate(x_test,y_test)