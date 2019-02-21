# -*- coding: utf-8 -*-
"""
Created on Sat Jan 26 15:42:33 2019

@author: Administrator
"""

import Knn
import os
import numpy as np
from PIL import Image

def loadImage(filename):
    
    # 读取图片
    im = Image.open(filename)

    # 显示图片
    #im.show() 
    
    im = im.convert("L") 
    
    data = im.getdata()
    data = np.array(data,float)
    data[data < 255]=1
    data[data == 255]=0
    return data

def classhandwrite(data):
    hwLabels = []
    trainingFileList = os.listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:] = Knn.img2vector('trainingDigits/%s'%fileNameStr)
    classifyresult = Knn.classify0(data,trainingMat,hwLabels,3)
    print("result:%d"%classifyresult)
    
if __name__=='__main__':
    data = loadImage('test/2_1.png')
    classhandwrite(data)
