import pandas as pd
import numpy as np

'''dataset = pd.read_csv('C:/users/inwin/downloads/DWH_Training.csv',
                      names = ['Height','Weight','Gender'])
dataset1 = pd.read_csv('C:/users/inwin/downloads/DWH_Test.csv',
                      names = ['Height','Weight','Gender', 'XX'])
height = np.array(dataset.Height)
weight = np.array(dataset.Weight)
org_gender = np.array(dataset.Gender)
height_test = np.array(dataset1.Height)
weight_test = np.array(dataset1.Weight)
test_gender = np.array(dataset1.Gender)
'''
def sigmoid(x):
	return 1.0/(1+ np.exp(-x))

def sigmoid_derivative(x):
	return x * (1.0 - x)

class NeuralNetwork:
    def __init__(self, x, y):
        self.input      = x
        self.w1=np.random.rand(self.input.shape[1],5) 
        self.w2=np.random.rand(5,5)
        self.w3=np.random.rand(5,1)                 
        self.y=y
        self.output=np.zeros(self.y.shape)
    def feedforward(self):
    	self.layer1=sigmoid(np.dot(self.input,self.w1))
    	self.layer2=sigmoid(np.dot(self.layer1,self.w2))
    	self.output=sigmoid(np.dot(self.layer2,self.w3))
    	print(self.layer1)
    	print(self.layer2)
    	print(self.output)

X = np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])
y = np.array([[0],[1],[1],[0]])	

nn=NeuralNetwork(X,y)
nn.feedforward()
'''for i in range(1500):
nn.feedforward()
'''