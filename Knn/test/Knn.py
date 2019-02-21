# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 19:00:19 2019

@author: Administrator
"""
from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1))-dataSet
    sqdiffMat = diffMat**2
    sqdistances = sqdiffMat.sum(axis=1)
    distances = sqdistances**0.5
    sortedDistIndicies = distances.argsort()#排序下标

    classCount={}
    #对labels进行汇总classCount={'A':2,'B':1}
    for i in range(k):
        voteilabels = labels[sortedDistIndicies[i]]
        classCount[voteilabels]=classCount.get(voteilabels,0)+1
    sortedclasscount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)#sortedclasscount=[('A',2),('B',1)]
    return sortedclasscount[0][0]

def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index=0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
        
    return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals,(m,1))
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

def datingClassTest():
    hoRatio = 0.1
    datingDatMat,datingLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDatMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print("the classifi came back:%d,the real answer is %d"%(classifierResult,datingLabels[i]))
        
        if (classifierResult!= datingLabels[i]):errorCount+=1.0
    print("the total error is %f"%(errorCount/float(numTestVecs)))
    
def classifyperson():
    resultList = ['not at all','in small doses','in large doses']
    percenttats = float(input("spent playing game?"))
    ffmile = float(input("ffmile?"))
    icecream = float(input("icecream?"))
    
    datingDatMat,datingLabels = file2matrix("datingTestSet2.txt")
    normMat,ranges,minVals = autoNorm(datingDatMat)
    inarr = array([ffmile,percenttats,icecream])
    classifierResult = classify0((inarr-minVals)/ranges,normMat,datingLabels,3)
    
    print(resultList[classifierResult-1])
    