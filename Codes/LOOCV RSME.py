#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import csv
import math
import matplotlib.pyplot as plt
from random import shuffle
import random
import time


# In[2]:


data = pd.read_csv('Metro_Interstate_Traffic_Volume.csv')


# In[3]:


data['holiday'] = pd.Categorical(data['holiday']).codes
data['weather_main'] = pd.Categorical(data['weather_main']).codes
data['weather_description'] = pd.Categorical(data['weather_description']).codes
data['year'] = data['date_time'].map(lambda x: int(x.split(" ")[0].split("-")[0]))
data['month'] = data['date_time'].map(lambda x: int(x.split(" ")[0].split("-")[1]))
data['day'] = data['date_time'].map(lambda x: int(x.split(" ")[0].split("-")[2]))
data['hour'] = data['date_time'].map(lambda x: int(x.split(" ")[1].split(":")[0]))
data.drop(['date_time'], axis=1, inplace=True)


# In[4]:


y = data['traffic_volume'].copy(deep=True)
X = data.copy(deep=True)
X.drop(['traffic_volume'], inplace=True, axis=1)


# In[5]:


def normalize(inp):
    minimum = min(inp)
    maximum = max(inp)
    m = maximum - minimum
    out = (inp - minimum)/ m
    output = np.array(out)
    return output


# In[6]:


for column in X:
    X[column] = normalize(X[column])
y = normalize(y)


# In[7]:


length = len(X)
train_data = X[:int(0.95 * length)]
test_data = X[int(0.95 * length) :]
train_result = y[:int(0.95 * length)]
test_result = y[int(0.95 * length) : ]


# In[8]:


def fit(X,y,c):
    
    inverse = np.linalg.inv(X.T @ X + (np.identity(X.shape[1])/(2*c)))
    ridge_regression = inverse @ X.T @ y
   
    return inverse, ridge_regression
   


# In[9]:


def predict(X,rr):
    pred = np.dot(X,rr)
    return pred


# In[10]:


def error(inver, rr, x, y):
    err_num = y - np.dot(rr.T, x.T)
    par_den = np.dot(x, inver)
    err_den = 1 - np.dot(par_den, x.T)
    err_term = np.divide(err_num,err_den)
    
    return err_term


# In[11]:


def loocv(X_train, y_train, c):
    num = 0
    total = 0
    j = len(X_train)
    for k in range(1,j):
        lcv_test = (X_train[k-1:k])
        X_train = (X_train[1:j])
        y_test = (y_train[k-1:k])
        y_train = (y_train[1:j])
        inv, RR = fit (X_train, y_train, c)
        lcv_err = error(inv, RR, lcv_test, y_test)
        total += lcv_err
        num += 1
    total = total/num
    return total


# In[12]:


C = [0.01,0.1,1,10,100]
best_error = 1
best_C = 0
for i in C:
    lcv = loocv(train_data, train_result,i)
    err = np.sqrt(np.square(lcv))
    print('The error for', i , 'is' , err)
    if err < best_error:
        best_error = err
        best_C = i
print("Best C", best_C)


# In[13]:


inv, rr = fit(train_data, train_result, 0.1)
predicted = predict(test_data, rr)


# In[14]:


deviation1 = predicted - test_result


# In[15]:


deviation2 = (predicted - test_result)**2


# In[16]:


plt.boxplot(deviation1)


# In[17]:


plt.boxplot(deviation2)


# In[ ]:




