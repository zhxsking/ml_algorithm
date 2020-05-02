# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 09:37:09 2018

@author: sking
"""

import numpy
import matplotlib.pyplot as plt

# 读取数据
def readData(filename):
    data = []
    label = []
    fr = open(filename)
    for line in fr.readlines():
        lineArr = line.strip().split()
        #前面的1.0，表示方程的常量。比如两个特征X1,X2，共需要三个参数，W1+W2*X1+W3*X2
        data.append([1.0, float(lineArr[0]), float(lineArr[1])])
        label.append(int(lineArr[2]))
    return data, label

# sigmoid函数
def sigmoid(z):
    return 1.0 / (1 + numpy.exp(-z))

# 梯度下降法，alpha为迭代步长，iterNum为迭代次数
def gradDescent(data, label, alpha, iterNum):
    dataMat = numpy.mat(data)
    labelMat = numpy.mat(label).T
    m, n = numpy.shape(dataMat)
    weights = numpy.ones((n,1)) #初始化
    for k in range(iterNum):
        h = sigmoid(dataMat * weights) #损失函数
        E = (h - labelMat)
        weights = weights - alpha * dataMat.T * E #更新权重，此处的dataMat.T*E即为推导后的梯度表达式
        alpha = 4 / (1 + k) + 0.01 #动态步长
    return weights

# 画出分类图
def plotLine(data, label, weights):
    dataArr = numpy.array(data)
    n = numpy.shape(dataArr)[0]
    pot1_x = []; pot1_y = []
    pot2_x = []; pot2_y = []
    for i in range(n):
        if int(label[i])== 1:
            pot1_x.append(dataArr[i,1])
            pot1_y.append(dataArr[i,2])
        else:
            pot2_x.append(dataArr[i,1])
            pot2_y.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(pot1_x, pot1_y, s=25, c='red', marker='x')
    ax.scatter(pot2_x, pot2_y, s=25, c='green')
    # 画线
    x = numpy.arange(-4.0, 4.0, 0.1)
    y = (-weights[0] - weights[1] * x) / weights[2]
    ax.plot(x, y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

filename = '.\\testSet.txt' #文件目录
data, label = readData(filename)
weights = gradDescent(data, label, 0.01, 1000).getA()
plotLine(data, label, weights)
