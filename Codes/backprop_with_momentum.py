#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
train = pd.read_csv('DWH_Training.csv', names = ['Height', 'Weight', 'Gender'])
train = train.replace(to_replace = -1, value = 0)


# In[2]:


def normalize(data):
    min_height = min(data['Height'])
    max_height = max(data['Height'])
    min_weight = min(data['Weight'])
    max_weight = max(data['Weight'])
    data['Height'] = (data['Height'] - min_height)/(max_height - min_height)
    data['Weight'] = (data['Weight'] - min_weight)/(max_weight - min_weight)
    x = np.array(data[['Height','Weight']])
    y = np.array([[i] for i in data.Gender])
    return x,y


# In[3]:


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


# In[4]:


def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# In[5]:


class ANN:
    def __init__(self,x,y):
        self.input = x
        self.w1 = np.random.rand(self.input.shape[1],5)
        self.w2 = np.random.rand(5,5)
        self.w3 = np.random.rand(5,1)
        self.y=y
        self.output = np.zeros(self.y.shape)
        self.b1=np.random.rand(self.input.shape[0],5)
        self.b2=np.random.rand(self.input.shape[0],5)
        self.b3=np.random.rand(self.input.shape[0],1)        
        self.loss = 0
        self.m1 = np.zeros((2,5))
        self.m2 = np.zeros((5,5))
        self.m3 = np.zeros((5,1))
        self.m_b1 = np.zeros((1,5))
        self.m_b2 = np.zeros((1,5))
        self.m_b3 = np.zeros((1,1))
    def loss_function(self):
        self.output[self.output==1]=0.999873
        self.output[self.output==0]=0.000901
        return -(self.y/self.output)+((1-self.y)/(1-self.output))
    def feedforward(self):
        self.hidden_layer1 = sigmoid(np.dot(self.input,self.w1)+self.b1)
        self.hidden_layer2 = sigmoid(np.dot(self.hidden_layer1,self.w2)+self.b2)
        self.output = sigmoid(np.dot(self.hidden_layer2,self.w3)+self.b3)   
    def back_propogation(self,c,l):
        loss =[]
        iteration =[]
        for t in range(1,10000):
            self.feedforward()
            loss_func= self.loss_function()
            delta_3=np.multiply(loss_func,d_sigmoid(self.output))
            d_weights3 = np.dot(self.hidden_layer2.T,delta_3)
            delta_2=np.dot(delta_3,self.w3.T)*d_sigmoid(self.hidden_layer2)
            d_weights2 = np.dot(self.hidden_layer1.T,delta_2)
            delta_1=np.dot(delta_2,self.w2.T)*d_sigmoid(self.hidden_layer1)
            d_weights1 = np.dot(self.input.T,delta_1)
            d_bias3=np.multiply(loss_func,d_sigmoid(self.output))
            d_bias2=d_bias3*d_sigmoid(self.hidden_layer2)
            d_bias1=d_bias2*d_sigmoid(self.hidden_layer1) 
            self.m1 = (c * self.m1) + d_weights1
            self.m2 = (c * self.m2) + d_weights2
            self.m3 = (c * self.m3) + d_weights3
            self.m_b1 = (c * self.m_b1) + d_bias1
            self.m_b2 = (c * self.m_b2) + d_bias2
            self.m_b3 = (c * self.m_b3) + d_bias3
            self.w3-=l/(len(self.input)) * self.m3
            self.w2-=l/(len(self.input)) * self.m2
            self.w1-=l/(len(self.input)) * self.m1
            self.b3-=l/(len(self.input)) * self.m_b3
            self.b2-=l/(len(self.input)) * self.m_b2
            self.b1-=l/(len(self.input)) * self.m_b1
            nn.loss=np.mean(np.square((nn.y*np.log(nn.output))+((1-nn.y)*np.log(1-nn.output))))
            if t%250 == 0:
                iteration.append(t)
                loss.append(nn.loss)
        #print(iteration, loss)
        return iteration,loss


# In[6]:


X,y=normalize(train)
nn=ANN(X,y)
sig = [0.1, 0.5, 0.9]
l = [0.1,10,100]
for i in sig:
    for j in l:
        plot_x, plot_y = nn.back_propogation(i,0.06)
        plt.plot(plot_x, plot_y)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title(i)
        plt.show()


# In[ ]:




