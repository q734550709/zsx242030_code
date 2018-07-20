#回归预测
#美国波士顿房价预测
#使用随机梯度下降
#波士顿房价数据读取器
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
#导入用于回归性能的评估
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import numpy as np
boston=load_boston()
print(boston.DESCR)

X=boston.data
y=boston.target

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=33,test_size=0.25)

#分析回归目标值的差异

print("The max target value is",np.max(boston.target))
print("The min target value is",np.min(boston.target))
print("The average target value is",np.average(boston.target))

ss_X=StandardScaler()
ss_y=StandardScaler()

X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)

y_train=ss_y.fit_transform(y_train)
y_test=ss_y.transform(y_test)

#线性回归
lr=LinearRegression()
lr.fit(X_train,y_train)
lr_y_predict=lr.predict(X_test)

#梯度下降
sgdr=SGDRegressor()
sgdr.fit(X_train,y_train)
sgdr_y_predict=sgdr.predict(X_test)

#性能测评指标
#平均绝对误差(Mean Absolute Error,MAN)
#均方误差(Mean Squared Error,MSE)
#R-squared
#这也是线性回归模型所要优化的目标

print("The value of default measurement of LinearRegression is",lr.score(X_test,y_test))

#使用r2_score模块,并输出评估结果
print("The value of R-squared of LinearRegression is",r2_score(y_test,lr_y_predict))

#使用mean_squared_error模块,并输出评估结果
print("The mean squared error of LinearRegression is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))

#使用mean_absolute_error模块,并输出评估结果
print("The mean absolute error of LinearRegression is",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(lr_y_predict)))

print("==============================")
#SGDRegressor
print("The value of default measurement of SGDRegressor is",sgdr.score(X_test,y_test))

#使用r2_score模块,并输出评估结果
print("The value of R-squared of SGDRegressor is",r2_score(y_test,sgdr_y_predict))

#使用mean_squared_error模块,并输出评估结果
print("The mean squared error of SGDRegressor is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))

#使用mean_absolute_error模块,并输出评估结果
print("The mean absolute error of SGDRegressor is",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(sgdr_y_predict)))

#在不清楚特征之间关系的前提下,我们任然可以使用线性回归作为大多数科学实验的基线系统


























































































































































