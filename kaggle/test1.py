import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression

#画出数据的分布
df_train=pd.read_csv(r'C:\Users\zsx\Desktop\kaggle\breast-cancer-train.csv')
df_test=pd.read_csv(r'C:\Users\zsx\Desktop\kaggle\breast-cancer-test.csv')
df_test_negative=df_test.loc[df_test['Type']==0][['Clump Thickness','Cell Size']]
df_test_positive=df_test.loc[df_test['Type']==1][['Clump Thickness','Cell Size']]
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')
plt.xlabel("Clump Thickness")
plt.ylabel("Cell Size")
plt.show()

#自己生成一个随机直线
#生成一个随机数
intercept=np.random.random([1])
print(intercept)
#生成两个随机数
coef=np.random.random([2])
print(coef)
lx=np.arange(0,12)
ly=(-intercept-lx*coef[0])/coef[1]
plt.plot(lx,ly,c='yellow')

#用随机生成的直线分割数据
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')
plt.xlabel("Clump Thickness")
plt.ylabel("Cell Size")
plt.show()

#使用线性回归线训练10个数据,生成直线
lr=LogisticRegression()
lr.fit(df_train[['Clump Thickness','Cell Size']][:10],df_train['Type'][:10])
print("Testing accuracy(10 traing samples):",lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type']))

#画出线性回归生成的数据分割
intercept=lr.intercept_
coef=lr.coef_[0,:]
ly=(-intercept-lx*coef[0])/coef[1]
plt.plot(lx,ly,c='green')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')
plt.xlabel("Clump Thickness")
plt.ylabel("Cell Size")
plt.show()

#训练全部的数据生成直线
lr=LogisticRegression()
lr.fit(df_train[['Clump Thickness','Cell Size']],df_train['Type'])
print("Testing accuracy(10 traing samples):",lr.score(df_test[['Clump Thickness','Cell Size']],df_test['Type']))

intercept=lr.intercept_
coef=lr.coef_[0,:]
ly=(-intercept-lx*coef[0])/coef[1]
plt.plot(lx,ly,c='blue')
plt.scatter(df_test_negative['Clump Thickness'],df_test_negative['Cell Size'],marker='o',s=200,c='red')
plt.scatter(df_test_positive['Clump Thickness'],df_test_positive['Cell Size'],marker='x',s=150,c='black')
plt.xlabel("Clump Thickness")
plt.ylabel("Cell Size")
plt.show()






















