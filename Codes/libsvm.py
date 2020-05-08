#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.colors as mcolors
from liblinear import *
from liblinearutil import *


dataset = pd.read_csv('/home/ajaychawda58/Desktop/ML Exercise/DWH_Training.csv', names = ['Height', 'Weight','Gender'])


height = np.array(dataset.Height)
weight = np.array(dataset.Weight)
org_gender = np.array(dataset.Gender)
length = int(len(height))

liblin_str = []
for i in range(length):
    liblin_str.append(str(org_gender[i])+ " " + "1:"+str(height[i])+ " " + "2:"+str(weight[i]))


# In[9]:


file = open("DWH_Training.txt", "a+")
for i in range(length):
    file.write(liblin_str[i] + "\n")
  
file.close()


# In[10]:


gender = dataset.Gender
color = np.array(['red','blue'])
plt.scatter(dataset.Height, dataset.Weight, c = gender, 
            cmap = mcolors.ListedColormap(['blue','red']))


# In[13]:


dataset1 = pd.read_csv('/home/ajaychawda58/Desktop/ML Exercise/DWH_test.csv', names = ['Height', 'Weight','Gender','XX'])

height = np.array(dataset1.Height)
weight = np.array(dataset1.Weight)
org_gender = np.array(dataset1.Gender)
length = int(len(height))
test_str = []
for i in range(length):
    test_str.append(str(org_gender[i])+ " " + "1:"+str(height[i])+ " " + "2:"+str(weight[i]))


# In[14]:


file = open("DWH_Test.txt", "a+")
for i in range(length):
    file.write(test_str[i] + "\n")
  
file.close()


# In[15]:Use Liblinear Tool


y, x = svm_read_problem('DWH_Training.txt')
y1, x1 = svm_read_problem('DWH_Test.txt')
param = parameter('-s 2 -B 10')
prob = problem(y, x)
m = train(prob,param)
p_labs, p_acc, a_vals = predict(y1,x1,m)
print(p_acc)


# In[16]:


[W,b] = m.get_decfun()
print(W,b)
a = W[0]/W[1]
xx = np.linspace(min(dataset.Height), max(dataset.Height))
yy = -a * xx - b/ W[1]
h0 = plt.plot(xx,yy, "k-", label="non weighted")
plt.scatter(dataset.Height, dataset.Weight, c = gender, 
            cmap = mcolors.ListedColormap(['blue','red']))


# In[ ]:




