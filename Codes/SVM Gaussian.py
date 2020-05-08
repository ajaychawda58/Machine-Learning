#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import math


# In[2]:


train_data = pd.read_csv("EX7data1.csv", names = ['index', 'feature_1', 'feature_2', 'class'])
test_data = pd.read_csv("EX7data2.csv", names = ['index', 'feature_1', 'feature_2', 'class'])


# In[3]:


train_data = train_data[['feature_1','feature_2','class']]
test_data = test_data[['feature_1','feature_2','class']]


# In[4]:


random_values = [math.pow(10,-6),math.pow(10,-5),math.pow(10,-4),math.pow(10,-3),math.pow(10,-2),math.pow(10,-1),math.pow(10,1),math.pow(10,2),math.pow(10,3),math.pow(10,4),math.pow(10,5),math.pow(10,6)]
sigma_values = []


# In[5]:


sigma_values += [i * 5 for i in random_values]
sigma_values += [i/4 for i in sigma_values]
sigma_values.sort()


# In[6]:


gamma = [np.sqrt(i) for i in sigma_values]


# In[7]:


X_train = train_data[['feature_1','feature_2']]
X_test = test_data[['feature_1','feature_2']]
Y_train = train_data['class']
Y_test = test_data['class']


# In[8]:


train_plot = []
test_plot = []


# In[9]:


def plot_graph():
    plt.plot(X_axis, train_plot, color = 'b', label = 'train')
    plt.plot(X_axis, test_plot, color = 'r', label = 'test')
    plt.text(6,90, 'overfit', color = 'g')
    plt.text(-6,67, 'underfit', color = 'g')
    plt.text(1.5,87, 'bestfit', color = 'g')
    plt.legend(loc = 0 )
    plt.xlabel('Sigma in log 10')
    plt.ylabel('Accuracy')
    plt.show()


# In[10]:


for i in gamma:
    svm = SVC(kernel = 'rbf', gamma = i)
    svm.fit(X_train, Y_train)
    T_predict = svm.predict(X_train)
    T_acc = np.sum(T_predict == Y_train)/ len(Y_train) * 100
    t_predict = svm.predict(X_test)
    t_acc = np.sum(t_predict == Y_test)/ len(Y_test) * 100
    train_plot.append(T_acc)
    test_plot.append(t_acc)
    X_axis = [np.log10(i) for i in sigma_values]





# In[13]:


plot_graph()


# In[ ]:




