#无监督学习在海量数据处理中是非常实用的技术
#数据聚类
#最为经典并且易用的聚类模型当属K均值算法（K-means）
#该算法要求我们预先设定聚类的个数,然后不断更新聚类中心,经过几轮这样的迭代,最后的目标就是要让所有数据点到其所属聚类中心距离的平方和趋于稳定。
#=====================================K均值算法===============================================
#http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/
#手写数字图像分为两个数据集合
#训练集3823条数据
#测试数据1797条数据
#图像通过8*8的像素矩阵,共有64个像素维度
#1个目标维度用来标记每个图像样本代表的数字类别
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist

digits_train=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",header=None)
digits_test=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",header=None)

#digits_train=pd.read_csv(r"C:\Users\zsx\Desktop\kaggle\optdigits.tra",header=None)
#digits_test= pd.read_csv(r"C:\Users\zsx\Desktop\kaggle\optdigits.tes",header=None)

#从训练与测试数据集上度分离出64维度的像素特征与1维度的是数字目标
#取出前64列的数据作为特征
X_train=digits_train[np.arange(64)]
#取出最后一列的数据作为标记
y_train=digits_train[64]

X_test=digits_test[np.arange(64)]
y_test=digits_test[64]

#从sklearn.cluster中导入KMeans模型

#初始化KMeans模型,并设置聚类中心数量为10
kmeans=KMeans(n_clusters=10)
kmeans.fit(X_train)
#逐条判断每个测试图像所属的聚类中心
y_pred=kmeans.predict(X_test)

#性能评估使用Adjusted Rand Index(ARI)
print(metrics.adjusted_rand_score(y_test,y_pred))

#如果被用于评估的数据没有所属类别,那么我们习惯使用轮廓系数来度量聚类结果的质量
#下面说明轮廓系数和聚类效果的关系
plt.subplot(3,2,1)
x1=np.array([1,2,3,1,5,6,5,5,6,7,8,9,7,9])
x2=np.array([1,3,2,2,8,6,7,6,7,1,2,1,1,3])
X=np.array(np.c_[x1,x2]).reshape(len(x1),2)
print(X)

#在1号子图做出原始数据点阵的分布
plt.xlim([0,10])
plt.ylim([0,10])
plt.title("Instances")
plt.scatter(x1,x2)

colors=['b','g','r','c','m','y','k','b']
markers=['o','s','D','v','^','p','*','+']

clusters = [2, 3, 4, 5, 8]
subplot_counter = 1
sc_scores = []
for t in clusters:
    subplot_counter += 1
    plt.subplot(3, 2, subplot_counter)
    kmeans_model = KMeans(n_clusters=t).fit(X)
    for i, l in enumerate(kmeans_model.labels_):
        plt.plot(x1[i], x2[i], color=colors[l], marker=markers[l], ls='None')
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    sc_score = silhouette_score(X, kmeans_model.labels_, metric='euclidean')
    sc_scores.append(sc_score)

# 绘制轮廓系数与不同类簇数量的直观显示图。
    plt.title('K = %s, silhouette coefficient= %0.03f' %(t, sc_score))
    
# 绘制轮廓系数与不同类簇数量的关系曲线。
plt.figure()
plt.plot(clusters, sc_scores, '*-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')

plt.show()
#我们得知当聚类中心数量为3的时候,轮廓系数最大。
#具有两大缺陷：
#（1）容易收敛到局部最优解
#（2）需要预先设定簇的数量
#可以使用肘部法用于粗略的估计合理的类簇个数
#当斜率区域稳定的时候就是K值



#肘部观察法实例
#使用均匀分布函数随机三个类簇,每个类簇周围10个数据样本
cluster1=np.random.uniform(0.5,1.5,(2,10))
cluster2=np.random.uniform(5.5,6.5,(2,10))
cluster3=np.random.uniform(3.0,4.0,(2,10))
print(cluster1)
print(cluster2)
print(cluster3)

#绘制30个数据样本的分布图像
X=np.hstack((cluster1,cluster2,cluster3)).T
print(X)
print(X.shape)
plt.scatter(X[:,0],X[:,1])
plt.xlabel('x1')
plt.ylabel('x2')
plt.show()

#测试9中不同聚类中心数量下,每种情况的剧烈质量,并作图
K=range(1,10)
meandistortions=[]

for k in K:
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(X)
    meandistortions.append(sum(np.min(cdist(X,kmeans.cluster_centers_,'euclidean'),axis=1))/X.shape[0])
plt.plot(K,meandistortions,'bx-')
plt.xlabel('K')
plt.ylabel('Average Dispersion')
plt.title("Selecting k with the Elbow Method")
plt.show()










































































