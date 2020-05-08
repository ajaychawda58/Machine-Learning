#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolor
import math
import time
dataset = pd.read_csv('BIG_DWH_Training.csv', names = ['Height','Weight','Gender'])


# In[2]:


height = np.array(dataset.Height)
weight = np.array(dataset.Weight)
org_gender = np.array(dataset.Gender)
index = dataset.index
length = int(len(height))


# In[3]:


def get_gradient(weight_bias,C,sample,sample_index):
    grad_b = weight_bias[1]
    grad_w = weight_bias[0]
    y = sample.Gender
    x = [sample.Height, sample.Weight]
   
    for k in sample_index:
        if(y[k] * (grad_w[0]*x[0][k] + grad_w[1]*x[1][k] + grad_b)) > 1:
            weight_bias = weight_bias
      
        else:
           
            weight_bias[0][0] += -(C * (y[k]*x[0][k]))
            weight_bias[0][1] += -(C * (y[k]*x[1][k]))
            weight_bias[1] += -(C * (y[k]))
         
    return (weight_bias)


# In[6]:



def gradient_descent(dataset,C, B):
    for j in range(1,10001):       
        sample = dataset.sample(B)
        sample_index = sample.index
        new_weight_bias = get_gradient(weight_bias, C , sample, sample.index)
        weight_bias[0][0] = weight_bias[0][0] - ((weight_bias[0][0]* (float(1/j)) +  ((float(1/j))* new_weight_bias[0][0])))
        weight_bias[0][1] = weight_bias[0][1] - ((weight_bias[0][1]* (float(1/j)) +  ((float(1/j))* new_weight_bias[0][1])))
        weight_bias[1] = weight_bias[1] - ((float(1/j))* new_weight_bias[1]) 
    return(weight_bias)


# In[7]:


for i in range(10):
    if i == 0:
        start_time = time.time()
    train_set = dataset
   
    C =[0.1]
    B = [10]
    w_init = [0.0,0.0]
    b_init = 0.0
    best_accuracy = 0.0
    for k in C:
        for l in B:
            weight_bias = [w_init,b_init]
            [W,bb] = gradient_descent(train_set,k,l)
    end_time = time.time()

        


# In[9]:


print("average execution time = " + str((end_time - start_time)))


# In[ ]:




