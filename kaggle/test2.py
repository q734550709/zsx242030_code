#监督学习分为分类学习和回归预测
#分类学习：二分类和多分类
#线性分类器
#这个模型通过累加计算每个维度的特征与各自权重的乘积来帮助类别决策
#f(w,x,b)=W^T*X+b
#g(Z)=sigmoid(Z) #逻辑回归
#当z=0的时候,g(Z)=0.5
#当z>0.5时,g(Z)>0,被归为一类
#当z<0.5时,g(Z)<0,被归为一类
#这个模型可以在这组训练集上得到最大似然估计的概率
#使用随机梯度算法得到w和b

#随机梯度上升(SGA)和随机梯度下降(SGD)
#梯度上升用于目标最大化
#梯度下降用于目标最小化
#样本的数据
#1.Sample code number                    id number 
#2.Ciump Thickness                       1-10
#3.Uniformity of Cell Size               1-10
#4.Uniformity of Cell Shape              1-10
#5.Marginal Adhesion                     1-10
#6.Single Epithelial Cell Size           1-10  
#7.Bare Chromatin                        1-10
#8.Bland Chromatin                       1-10
#9.Normal Nucleoli                       1-10
#10.Mitoses                              1-10
#11.Class                                (2 for benign,4 for malignant)

import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
#创建特征列表
column_names=['Sample code number','Ciump Thickness','Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion','Single Epithelial Cell Size','Bare Chromatin','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
#使用pandas.read_csv函数从互联网读取指定数据
#data=pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=column_names)
data=pd.read_csv(r'C:\Users\zsx\Desktop\kaggle\breast-cancer-wisconsin.data',names=column_names)
print(data[:3])
#将?替换为标准缺失值表示
data=data.replace(to_replace='?',value=np.nan)
#丢弃有缺失值的数据
data=data.dropna(how='any')
print(data.shape)

X_train,X_test,y_train,y_test=train_test_split(data[column_names[1:10]],data[column_names[10]],test_size=0.25,random_state=33)
print(y_train.value_counts())
print(y_test.value_counts())

#对以上数据进行学习,并根据测试样本特征进行预测
#标准化数据,暴增每个维度的特征数据的方差为1,均值为0,使得预测结果不会被某些维度过大的特征值而主导
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

#初始化LogisticRegression与SGDClassifier
lr=LogisticRegression()
sqdc=SGDClassifier()

lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)

sqdc.fit(X_train,y_train)
sqdc_y_predict=sqdc.predict(X_test)


#肿瘤被分为：
#True Positive(真阳性)
#True Negative(真阴性)
#False Positive(假阳性)
#False Negative(假阴性)
#评判除了使用准确性（Accuracy）
#还使用召回率(Recall)和精确率(Precision)这两个评价指标
#它们的定义分别为：
#Accuracy=(#True Positive+#True Negative)/(#True Positive+#True Negative+#False Positive+#False Negative)
#Precision=#True Positive/(#True Positive+#True Negative)
#Recall=#True Positive/(#True Positive+#False Negative)

#为了综合考量召回率(Recall)和精确率(Precision)这两个评价指标,我们求这两个指标的调和平均数
#得到F1指标(F1 measure):
#F1 measure=2/(1/Precision+1/Recall)

#使用逻辑斯特回归准确性
print("Accuracy of LR Classifier:",lr.score(X_test,y_test))
print(classification_report(y_test,lr_y_predict,target_names=['Benign','Malignant']))

#使用梯度下降
print("Accuracy of SGD Classifier:",sqdc.score(X_test,y_test))
print(classification_report(y_test,sqdc_y_predict,target_names=['Benign','Malignant']))

#LR计算时间长但是准确性高
#SGD计算时间短但是准确性相对低
#10万量级以上的数据建议使用SGD










































































