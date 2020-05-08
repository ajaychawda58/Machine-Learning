#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
from math import sqrt
dataset = pd.read_csv('/home/ajaychawda58/Downloads/DWH_Training.csv',
                      names = ['Height','Weight','Gender'])
dataset1 = pd.read_csv('/home/ajaychawda58/Downloads/DWH_test.csv',
                      names = ['Height','Weight','Gender', 'XX'])
height = np.array(dataset.Height)
weight = np.array(dataset.Weight)
org_gender = np.array(dataset.Gender)
height_test = np.array(dataset1.Height)
weight_test = np.array(dataset1.Weight)
test_gender = np.array(dataset1.Gender)
length = int(len(dataset.Height))


#Nearest Centroid classifier function


def nearest_centroid (train_set, test_set):
    m_len= 0
    male_centroid_weight = 0
    male_centroid_height = 0
    female_centroid_height = 0

    f_len = 0
    female_centroid_weight = 0
    for i in range(length):

       if org_gender[i] == 1:
         m_len += 1
         male_centroid_height += height[i]
         male_centroid_weight += weight[i]
         plt.scatter(height[i], weight[i], c = 'blue', marker= '+' )
       else:
         f_len += 1
         female_centroid_height += height[i]
         female_centroid_weight += weight[i]
         plt.scatter(height[i], weight[i], c = 'red', marker= '_' )



    centroid_pos = [male_centroid_height/m_len, male_centroid_weight/m_len]
    centroid_neg = [female_centroid_height/f_len, female_centroid_weight/f_len]
    w  = 2 * np.subtract(centroid_neg,centroid_pos)
    b  = (centroid_pos[0]**2 + centroid_pos[1]**2) - (centroid_neg[0]**2 + centroid_neg[1]**2)
    plt.scatter(centroid_neg[0], centroid_neg[1], c = 'red', s = 100)
    plt.scatter(centroid_pos[0], centroid_pos[1], c = 'blue', s = 100)
    x = np.linspace(162,180, num = 100)

    y1 = - (b + w[0] * x)/w[1]
    plt.plot(x,y1)


    test_length = int(len(test_set.Height))
    accuracy = 0
    pred_gender = 0
    for i in range(test_length):

        dist_pos = sqrt((centroid_pos[0] - height_test[i])**2 + (centroid_pos[1] - weight_test[i])**2)
        dist_neg = sqrt((centroid_neg[0] - height_test[i])**2 + (centroid_neg[1] - weight_test[i])**2)
        if(dist_pos < dist_neg):

            if(test_gender[i] == 1):
                accuracy += 1
        else:

            if(test_gender[i] == -1):
                accuracy += 1

    #model accuracy
    check_accuracy = ((accuracy) * 100)/test_length
    print(' The accuracy of the nearest centroid classifier is :', check_accuracy)




# calling the ncc function
# dataset - Training Data
# dataset1 - Test Data

nearest_centroid(dataset, dataset1)


