#====================使用线性回归模型在比萨训练样本上进行拟合===========================
#输入训练样本的特征以及目标值,分别存储在变量X_train与y_train
X_train=[[6],[8],[10],[14],[18]]
y_train=[[7],[9],[13],[17.5],[18]]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn import feature_selection
from sklearn.cross_validation import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

#使用线性回归进行训练
regressor=LinearRegression()
regressor.fit(X_train,y_train)

xx=np.linspace(0,26,100)
xx=xx.reshape(xx.shape[0],1)

yy=regressor.predict(xx)

plt.scatter(X_train,y_train)
plt1,=plt.plot(xx,yy,label="Degree=1")
plt.axis([0,25,0,25])
plt.xlabel("Diameter of Pizza")
plt.ylabel("Piece of Pizza")
plt.legend(handles=[plt1])
#plt.show()

#输出线性回归模型在训练样本上的R-square值
print("The R-squared value of Linear Regressor performing on the training data is",regressor.score(X_train,y_train))

#使用2次多项式回归模型在比萨训练样本上进行拟合
#导入多项式特征产生器


poly2=PolynomialFeatures(degree=2)
X_train_poly2=poly2.fit_transform(X_train)

regressor_poly2=LinearRegression()
regressor_poly2.fit(X_train_poly2,y_train)

xx_poly2=poly2.transform(xx)

yy_poly2=regressor_poly2.predict(xx_poly2)

plt.scatter(X_train,y_train)
plt1,=plt.plot(xx,yy,label='Degree=1')
plt2,=plt.plot(xx,yy_poly2,label="Degree=2")

plt.axis([0,25,0,25])
plt.xlabel("Diameter if Pizza")
plt.ylabel("Price if Pizza")
plt.legend(handles=[plt1,plt2])
#plt.show()

print("The R-squared value of Linear Polynominal Regressor(Degree=2) performing on the training data is",regressor_poly2.score(X_train_poly2,y_train))

#使用4次多项式回归模型在比萨训练样本上进行拟合
poly4=PolynomialFeatures(degree=4)
X_train_poly4=poly4.fit_transform(X_train)

regressor_poly4=LinearRegression()
regressor_poly4.fit(X_train_poly4,y_train)

xx_poly4=poly4.transform(xx)

yy_poly4=regressor_poly4.predict(xx_poly4)

plt.scatter(X_train,y_train)
plt1,=plt.plot(xx,yy,label='Degree=1')
plt2,=plt.plot(xx,yy_poly2,label="Degree=2")
plt4,=plt.plot(xx,yy_poly4,label="Degree=4")
plt.axis([0,25,0,25])
plt.xlabel("Diameter if Pizza")
plt.ylabel("Price if Pizza")
plt.legend(handles=[plt1,plt2,plt4])
#plt.show()

print("The R-squared value of Linear Polynominal Regressor(Degree=4) performing on the training data is",regressor_poly4.score(X_train_poly4,y_train))
print("============================")
#评估3中回归模型在测试数据集上的性能表现
X_test=[[6],[8],[11],[16]]
y_test=[[8],[12],[15],[18]]
print(regressor.score(X_test,y_test))
X_test_poly2=poly2.transform(X_test)
print(regressor_poly2.score(X_test_poly2,y_test))
X_test_poly4=poly4.transform(X_test)
print(regressor_poly4.score(X_test_poly4,y_test))

#L1范数正则化,通常被称为Lasso
#正则化的目的在于提高模型在位置测试数据上的泛化能力,避免参数过拟合
#正则化的常见方法都是在原来模型优化目标的基础上,增加对参数的惩罚项

#Lasso模型在4次多项式特征上的拟合表现
lasso_poly4=Lasso()
lasso_poly4.fit(X_train_poly4,y_train)
print(lasso_poly4.score(X_test_poly4,y_test))
print(lasso_poly4.coef_)
print(regressor_poly4.score(X_test_poly4,y_test))
print(regressor_poly4.coef_)

#L2范数正则化,通常被称为Ridge
#Ridge模型在4次多项式特征上的拟合表现
print(regressor_poly4.coef_)
print(np.sum(regressor_poly4.coef_**2))
ridge_poly4=Ridge()
ridge_poly4.fit(X_train_poly4,y_train)
print(ridge_poly4.score(X_test_poly4,y_test))
print(ridge_poly4.coef_)
print(np.sum(ridge_poly4.coef_**2))




















































































































