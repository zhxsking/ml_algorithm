# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets

from kmeans import Kmeans
from knn import Knn


# 导入数据
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Kmeans
mdl_kmeans = Kmeans(k=3)
mdl_kmeans.fit(X)

# KNN
n_neighbors = 15
mdl_knn = Knn(k=n_neighbors)
mdl_knn.fit(X, y)

# 模型库
mdls = [mdl_kmeans, mdl_knn]

for mdl in mdls:
    # 绘制预测图
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))
    Z = mdl.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # 标出训练数据
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


plt.show()