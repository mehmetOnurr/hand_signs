# -*- coding: utf-8 -*-
"""
Created on Thu May 27 12:59:13 2021

@author: Asus
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
def initialize_weigths_and_bias(dimension):
    w = np.full((dimension,1), 0.01)
    b = 0.0
    return w,b

w,b = initialize_weigths_and_bias(4096)

#%% sigmoid fonc

def sigmoid(z):
    y_head = 1 / (1+ np.exp(-z))
    
    return y_head

y_head = sigmoid(0)
print(y_head)

#%% forward prop

def forward_propagation(w,b,x_train,y_train):
    z = np.dot(w.T,x_train)+b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head)-(1-y_train)* np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    return cost

#%% forward and backward prop

def forward_backward_propagation(w,b,x_train,y_train):
    # forward propagation
    z = np.dot(w.T,x_train)+b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head)-(1-y_train)* np.log(1-y_head)
    cost = (np.sum(loss))/x_train.shape[1]
    
    # backward propagation
    
    derivative_weight = (np.dot(x_train,((y_head - y_train).T)))/ x_train.shape[1]
    derivative_bias = np.sum(y_head - y_train)/ x_train.shape[1]
    
    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost, gradients

#%% update parameters


def update(w,b, x_train, y_train, learning_rate, number_of_iterarion):
    cost_list = []
    cost_list2 = []
    
    index = []
  
    for i in range(number_of_iterarion):    
        
        cost,gradients = forward_backward_propagation(w,b,x_train,y_train)
        cost_list.append(cost)
        
        w = w - learning_rate * gradients["derivative_weight"]
        b = b-  learning_rate * gradients["derivative_bias"]
        
        if i% 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print("Cost after iteration %i : %f"%(i,cost))
        
    parameters = {"weight" : w, "bias": b}
    plt.plot(index, cost_list2)
    plt.xticks(index,rotation ='vertical')
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()
        
    return parameters, gradients, cost_list
#%% predict func

def predict(w,b,x_test):
    z = sigmoid(np.dot(w.T,x_test)+b)
    
    Y_prediction =np.zeros((1,x_test.shape[1]))
    
    for i in range(z.shape[1]):
        if z[0,i]<= 0.5:
            Y_prediction[0,i]=0
        else:
            Y_prediction[0,i]=1
    return Y_prediction

#%% logictis regression

def logistic_regression(x_train,y_train,x_test,y_test,learning_rate,num_iterations):
    dimension = x_train.shape[0]
    
    w,b = initialize_weigths_and_bias(dimension)

    parameters , gradients , cost_list = update(w,b, x_train,y_train, learning_rate, num_iterations)

    y_prediction_test = predict(parameters["weight"], parameters["bias"],x_test)
    
    y_prediction_train = predict(parameters["weight"],parameters["bias"],x_train)
    
    print("train accuracy : {}".format(100- np.mean(np.abs(y_prediction_train))*100))
    
    print("test accuracy : {}".format(100- np.mean(np.abs(y_prediction_test))*100))
    
    

x_l = np.load("X.npy")
y_l = np.load("Y.npy")

img_size = 64
plt.subplot(1, 2, 1)
plt.imshow(x_l[260].reshape(img_size, img_size))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(x_l[900].reshape(img_size, img_size))
plt.axis('off')

X = np.concatenate((x_l[204:409], x_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]  

X_train_flatten = X_train.reshape(number_of_train,X_train.shape[1]*X_train.shape[2])
X_test_flatten = X_test .reshape(number_of_test,X_test.shape[1]*X_test.shape[2])
print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)


x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = Y_train.T
y_test = Y_test.T
print("x train: ",x_train.shape)
print("x test: ",x_test.shape)
print("y train: ",y_train.shape)
print("y test: ",y_test.shape)

logistic_regression(x_train,y_train,x_test,y_test,learning_rate=0.01,num_iterations=150)
    
    
#%% logistic reg izi

from sklearn import linear_model
logreg = linear_model.LogisticRegression(random_state=42)

print("Test accuracy {}".format(logreg.fit(x_train.T,y_train.T).score(x_test.T,y_test.T)))

print("train accuracy {}".format(logreg.fit(x_train.T,y_train.T).score(x_train.T,y_train.T)))
    
    
    
    
    
    
    
    
    
    
    
    
    
    