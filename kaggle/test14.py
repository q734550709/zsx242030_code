#特征降维（PCA）
#主成分分析(Principal Component Analysis)是最为经典和实用的降维特征技术
#特别是在图像识别方面有突出的表现

#=======================显示手写数字图片经PCA压缩后的二位空间的分布======================
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
M=np.array([[1,2],[2,4]])
#计算矩阵的侄
print(np.linalg.matrix_rank(M,tol=None))

digits_train=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",header=None)
digits_test=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",header=None)

X_digits=digits_train[np.arange(64)]
#取出最后一列的数据作为标记
y_digits=digits_train[64]

#初始化一个可以将高维度特征向量(64)压缩至二个维度的PCA
estimator=PCA(n_components=2)
X_pca=estimator.fit_transform(X_digits)
print(X_pca.shape)

def plot_pca_scatter():
    colors=['black','blue','purple','yellow','white','red','lime','cyan','orange','gray']
    for i in range(len(colors)):
        px=X_pca[:,0][y_digits.as_matrix()==i]
        py=X_pca[:,1][y_digits.as_matrix()==i]
        plt.scatter(px,py,c=colors[i])
    plt.legend(np.arange(0,10).astype(str))
    plt.xlabel("First Principal Component")
    plt.ylabel("Second Principal Component")
    plt.show()
plot_pca_scatter()


















































































