# -*- coding: utf-8 -*-
"""
Created on Wed Jan  9 14:29:12 2019

@author: zhxsking
"""

import numpy as np

epochs = 1000 # 循环次数
lr = 10 # 学习率

def relu(x, deriv=False):
    """Relu函数"""
    if(deriv==True):
        return np.where(x>0, 1, 0)
    return np.maximum(0, x)

def leakly_relu(x, deriv=False):
    """Leakly Relu函数"""
    emi = 0.2
    if(deriv==True):
        return np.where(x>0, 1, emi)
    return np.maximum(emi*x, x)

def sigmoid(x, deriv=False):
    """Sigmoid函数"""
    if(deriv==True):
        return (1 - x) * x # 导数
    return 1.0 / (1 + np.exp(-x))

if __name__ == '__main__':
    np.random.seed(1)
    # 建立异或数据集
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0, 1, 1, 0]]).T
    # 初始化权重
    weight1 = 2 * np.random.random((2, 4)) - 1
    weight2 = 2 * np.random.random((4, 1)) - 1
    # 指定激活函数
    active = sigmoid
    
    for i in range(epochs):
        # 前向传播
        layer1 = x
        layer2 = active(np.dot(layer1, weight1))
        layer3 = active(np.dot(layer2, weight2))
        # 计算误差，实际误差公式为均方误差（MSE），此处loss表达式为MSE求导后的结果，为反向传播做准备
        loss = layer3 - y
        mse = 0.5 * np.sum(loss**2)
        print(mse)
        # 根据链式法则反向传播
        error = (loss * active(layer3, True)) # 误差项
        weight2_delta = np.dot(layer2.T, error) # 链式法则
        layer2_loss = np.dot(error, weight2.T) # 根据误差项反推中间层对输出层loss的贡献率
        weight1_delta = np.dot(layer1.T, layer2_loss * active(layer2, True)) # 链式法则
        # 更新权重
        weight2 -= lr * weight2_delta
        weight1 -= lr * weight1_delta
    print(layer3)


