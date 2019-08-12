# -*- coding: utf-8 -*-

import numpy as np


class Kmeans():
    '''kmeans实现'''
    
    def __init__(self, k=2, num_iters=100):
        super().__init__()
        self.k = k
        self.num_iters = num_iters
        self.centroids = None
        self.clusters = None
        
    def _distance(self, x, y):
        return np.sqrt(np.sum((x-y)**2))
    
    def _get_init_centroids(self, data, k):
        return data[np.random.randint(0, len(data), k)]
    
    def _get_next_centroids(self, data, centroids_last):
        # 遍历求距离并计算类别
        for i in range(len(data)):
            best_dis = np.inf
            for j in range(self.k):
                dis = self._distance(data[i], centroids_last[j])
                
                if dis < best_dis:
                    best_dis = dis
                    self.clusters[i] = j
        
        # 计算新质心
        centroids_next = centroids_last.copy()
        for i in range(self.k):
            # 取出data中为第i类的数据并计算每一列均值作为新质心
            data_tmp = data[self.clusters == i]
            centroids_next[i] = np.mean(data_tmp, axis=0)
        
        return centroids_next     
    
    def fit(self, data):
        '''对数据进行拟合'''
        self.centroids = self._get_init_centroids(data, self.k)
        self.clusters = np.zeros(len(data))
        
        centroids_last = self.centroids
        for i in range(self.num_iters):
            centroids_new = self._get_next_centroids(data, self.centroids)
            
            if ((centroids_new == centroids_last).all()):
                print('stop in iter {}'.format(i+1))
                break
            
            centroids_last = centroids_new
            self.centroids = centroids_new
    
    def predict(self, x):
        # 遍历求距离并计算类别
        pred = np.zeros(len(x))
        for i in range(len(x)):
            best_dis = np.inf
            for j in range(self.k):
                dis = self._distance(x[i], self.centroids[j])
                
                if dis < best_dis:
                    best_dis = dis
                    pred[i] = j
        
        return pred
    

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from sklearn.datasets.samples_generator import make_blobs
    
    #np.random.seed(0)
    
    # 生成测试数据
    centers = [[1, 1], [-1, -1], [1, -1], [-1,1]]
    n_clusters = len(centers)
    X, lab = make_blobs(n_samples=1000, centers=centers, cluster_std=0.3)
    
    #lab[lab==1] = 0 # 非线性化
    
    mdl = Kmeans(k=4, num_iters=100)
    mdl.fit(X)
    pred = mdl.clusters
    predd = mdl.predict(X)
    
    plt.figure()
    plt.subplot(121)
    plt.scatter(X[:,0], X[:,1], s=15, c=lab)
    plt.subplot(122)
    plt.scatter(X[:,0], X[:,1], s=15, c=pred)
    
    
    
    
    