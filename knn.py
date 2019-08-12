# -*- coding: utf-8 -*-

import numpy as np
from collections import Counter


class Knn():
    '''KNN算法实现'''
    def __init__(self, k=1):
        super().__init__()
        self.k = k
        self._x_fit = None
        self._y_fit = None
        
    def _distance(self, a, b):
        return np.sqrt(np.sum((a-b)**2))
        
    def fit(self, x_fit, y_fit):
        self._x_fit = x_fit
        self._y_fit = y_fit
    
    def _predict(self, x):
        '''对单个数据进行预测'''
        distances = [self._distance(x, x_fit) for x_fit in self._x_fit]
        distances_idx = np.argsort(distances)
        y_top = [self._y_fit[a] for a in distances_idx[:self.k]] # 最近邻的标签
        votes = Counter(y_top) # 对标签计数
        
        return votes.most_common(1)[0][0] # 返回出现次数最多的
        
    def predict(self, x):
        return np.array([self._predict(_x) for _x in x])
        
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets.samples_generator import make_blobs
    
    #np.random.seed(0)
    
    # 生成测试数据
    centers = [[1, 1], [-1, -1], [1, -1], [-1,1]]
    n_clusters = len(centers)
    X, lab = make_blobs(n_samples=1000, centers=centers, cluster_std=0.5)
    
    lab[lab==1] = 0 # 非线性化
    
    mdl = Knn(k=15)
    mdl.fit(X, lab)
    pred = mdl.predict(X)
    
    plt.figure()
    plt.subplot(121)
    plt.scatter(X[:,0], X[:,1], s=15, c=lab)
    plt.subplot(122)
    plt.scatter(X[:,0], X[:,1], s=15, c=pred)
    