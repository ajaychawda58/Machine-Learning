#!/usr/bin/env python
# coding: utf-8

# In[84]:


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import math
dataset = pd.read_csv('C:/users/inwin/downloads/DWH_Training.csv',
                      names = ['Height','Weight','Gender'])


# In[85]:


mapIndexPosition = list(zip(dataset.Height, dataset.Weight, dataset.Gender))
np.random.shuffle(mapIndexPosition)
dataset.Height, dataset.Weight, dataset.Gender = zip(*mapIndexPosition)


# In[86]:


height = np.array(dataset.Height)
weight = np.array(dataset.Weight)
org_gender = np.array(dataset.Gender)
pred_gender = np.zeros(232)


# In[87]:


def dist_points(x1,y1,x2,y2):
    return math.sqrt( (x2 - x1)**2 + (y2 - y1)**2 );


# In[154]:


def predict(test_height,test_weight,k):
    distance = []

    for i in range(0,232):
          distance.append((dist_points(test_height,test_weight,height[i],weight[i])
                           ,org_gender[i]))
            
    distance.sort(key=lambda x: x[0])
    sum=0
    for i in range (0,k):
        sum=sum+ distance[i][1]
    return ( 1 if sum>=0 else -1)


# In[190]:


def k_nearest_neighbours(train_set,test_set,k,c):
  
    accuracy = 0
    size = np.shape(test_set)[0]
    slp = [0,23,46,69,92,115,138,161,184,207]
    
    for i in range(0,24):
        pred_gender = predict(height[i+slp[c]],weight[i+slp[c]],k)
        
        if org_gender[i+slp[c]] == pred_gender:
             accuracy += 1
            
        else:
             accuracy += 0
                    
    
    check_accuracy = ((accuracy) * 100)/size
    print(check_accuracy,k)       
   


# In[191]:


k = [3,5,20]
for i in range(1,11):
   test = height[int((i-1)*23):int((i)*23)]
   train = height[:int((i-1)*23)],height[int(i)*23:]
   
   for j in k:
       k_nearest_neighbours(train, test, j,(i-1))   


# In[ ]:




