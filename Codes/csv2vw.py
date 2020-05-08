#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np
dataset = pd.read_csv("INS_training.csv", usecols = [i for i in range(95)], header = None )
dataset = dataset[1:]
length = int(len(dataset))


# In[6]:


dataset[94] = dataset[94].replace({'Class_2' : 2, 'Class_3' : 3, 'Class_1' : 1, 'Class_4' : 4, 'Class_5' : 5, 'Class_6' : 6, 'Class_7' : 7, 'Class_8' : 8, 'Class_9' : 9})


# In[ ]:





# In[7]:



string = []
list_str = []
for i in range(length):
    
    string = (str(dataset[94][i+1]) + " " + "ex" + str(dataset[0][i+1]) + "|f "  )
    for j in range(1,94):
        if ((dataset[j][i+1]) != '0'):
            string = ( string + str(j) + ":" + str(dataset[j][i+1]) + " ")
    list_str.append(string)
file = open("/home/ajaychawda58/Desktop/ML Exercise/INS_Training.txt", "a+")
for i in list_str:
    file.write(i + "\n" )
file.close()


# In[8]:


dataset1 = pd.read_csv("INS_test.csv", usecols = [i for i in range(94)], header = None, )
dataset1 = dataset1[1:]
length = int(len(dataset1))


# In[9]:


string = []
list_str = []
for i in range(length):
    
    string = ("ex" + str(dataset1[0][i+1]) + "|f "  )
    for j in range(1,94):
        if ((dataset1[j][i+1]) != '0'):
            string = ( string + str(j) + ":" + str(dataset1[j][i+1]) + " ")
    list_str.append(string)
file = open("/home/ajaychawda58/Desktop/ML Exercise/INS_Test.txt", "a+")
for i in list_str:
    file.write(i + "\n" )
file.close()


# In[ ]:





# In[ ]:




