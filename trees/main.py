# -*- coding: utf-8 -*-
"""
Created on Sun Jan 27 16:08:27 2019

@author: Administrator
"""
import trees

if __name__=='__main__':
    data,labels = trees.createDataSet()
    myTree = trees.createTree(data,labels)
    