#使用Tensorflow自定义一个线性分类器用于对"良/恶性乳腺肿瘤"进行预测
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

train=pd.read_csv(r'C:\Users\zsx\Desktop\kaggle\breast-cancer-train.csv')
test= pd.read_csv(r'C:\Users\zsx\Desktop\kaggle\breast-cancer-test.csv')

#分割特征与分类目标
X_train=np.float32(train[['Clump Thickness','Cell Size']].T)
y_train=np.float32(train['Type'].T)
X_test=np.float32(test[['Clump Thickness','Cell Size']].T)
y_test=np.float32(test['Type'].T)

#定义一个tensorflow的变量b作为线性模型的截距,同时设置初始值为1.0
b=tf.Variable(tf.zeros([1]))
#设置初始值为-1.0至1.0之间均匀分布的随机数
W=tf.Variable(tf.random_uniform([1,2],-1.0,1.0))

y=tf.matmul(W,X_train)+b

#均方误差
loss=tf.reduce_mean(tf.square(y-y_train))

#迭代步长为0.01
optimizer=tf.train.GradientDescentOptimizer(0.01)

#最小二乘损失为优化为目标
train=optimizer.minimize(loss)

init=tf.initialize_all_variables()

sess=tf.Session()

sess.run(init)

for step in range(0,1000):
    sess.run(train)
    if step%200==0:
        print(step,sess.run(W),sess.run(b))
		
#准备测试样本
test_negative=test.loc[test['Type']==0][['Clump Thickness','Cell Size']]
test_positive=test.loc[test['Type']==1][['Clump Thickness','Cell Size']]
plt.scatter(test_negative['Clump Thickness'],test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(test_positive['Clump Thickness'],test_positive['Cell Size'],marker='x',s=150,c='black')
plt.xlabel("Clump Thickness")
plt.ylabel("Cell Size")

lx=np.arange(0,12)
ly=(0.5-sess.run(b)-lx*sess.run(W)[0][0])/sess.run(W)[0][1]
plt.plot(lx,ly,color='green')
plt.show()





























































