#K近邻(回归)
#配置不同的K近邻模型来比较回归性能的差异
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error

boston=load_boston()

X=boston.data
y=boston.target

X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=33,test_size=0.25)
ss_X=StandardScaler()
ss_y=StandardScaler()

X_train=ss_X.fit_transform(X_train)
X_test=ss_X.transform(X_test)

y_train=ss_y.fit_transform(y_train)
y_test=ss_y.transform(y_test)

dtr=DecisionTreeRegressor()
dtr.fit(X_train,y_train)
dtr_y_predict=dtr.predict(X_test)

print("===========================")

#使用r2_score模块,并输出评估结果
print("The value of R-squared of DecisionTreeRegressor is",dtr.score(X_test,y_test))
#使用mean_squared_error模块,并输出评估结果
print("The mean squared error of DecisionTreeRegressor is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dtr_y_predict)))
#使用mean_absolute_error模块,并输出评估结果
print("The mean absolute error of DecisionTreeRegressor is",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dtr_y_predict)))

#树模型可以解决非线性模型的问题
