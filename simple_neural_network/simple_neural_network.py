# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 17:09:27 2019

@author: zhxsking
"""

import numpy as np

epochs = 1000 # 训练次数
learn_rate = 0.1 # 学习率

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
    # 建立异或数据
    x = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y= np.array([[0, 1, 1, 0]]).T
    # 初始化权重，（-1，1）区间随机数矩阵
    weight1 = 2 * np.random.random((2,4)) - 1 # 输入层到中间层的权重
    weight2 = 2 * np.random.random((4,1)) - 1 # 中间层到输出层的权重
    # 定义激活函数
    active_fun = leakly_relu
    
    for i in range(epochs):
        # 数据填进网络，前向传播
        out1 = np.dot(x, weight1)
        out1_active = active_fun(out1)
        out2 = np.dot(out1_active, weight2)
        out2_active = active_fun(out2)
        # 计算loss
        loss = y - out2_active
        print(np.mean(np.abs(loss)))
        # 反向传播
        # 由于期望值为0或1，sigmoid函数值为0或1时导数最小，离0或1越远则导数越大；因此当输出值离0或1越远，则以越大的步子靠近0或1
        # 所以此处的点乘计算每一个样本在当前权重下的导数与loss的乘积，导数越大即需要以更大的步子靠近0或1，靠近哪一个以及往哪个方向靠近则由loss的大小以及loss的正负决定
        # 总而言之，loss指明真实值的方向，梯度让网络知道往这个方向迈多大的步子
        out2_delta = loss * active_fun(out2_active, deriv=True)
        out1_loss = np.dot(out2_delta, weight2.T) # 根据out2_delta反推中间层对输出层loss的贡献率
        out1_delta = out1_loss * active_fun(out1_active, deriv=True)
        # 更新参数，把权重增量依序加给原权重
        weight2 += np.dot(out1_active.T, out2_delta) * learn_rate
        weight1 += np.dot(x.T, out1_delta) * learn_rate
        
        
    print(out2_active)
    
    
    
    
    
    