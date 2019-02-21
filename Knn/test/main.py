# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 22:14:48 2019

@author: Administrator
"""
import Knn
import numpy as np
import matplotlib.pyplot as plt

datingDataMat,datingLables = Knn.file2matrix('datingTestSet2.txt')

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(datingDataMat[:,0],datingDataMat[:,1],15.0*np.array(datingLables),c=15.0*np.array(datingLables))

plt.show()
