#支持向量机(回归)
#使用三种不同核函数配置的支持向量机回归模型进行训练,并且分别对测试数据做出预测
from sklearn.svm import SVR
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split
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

#使用线性核函数
linear_svr=SVR(kernel='linear')
linear_svr.fit(X_train,y_train)
linear_svr_predict=linear_svr.predict(X_test)

#使用多项式核函数
poly_svr=SVR(kernel='poly')
poly_svr.fit(X_train,y_train)
poly_svr_predict=poly_svr.predict(X_test)

#使用径向基核函数
rbf_svr=SVR(kernel='rbf')
rbf_svr.fit(X_train,y_train)
rbf_svr_predict=rbf_svr.predict(X_test)

print("===========================")
#使用r2_score模块,并输出评估结果
print("The value of R-squared of linear SVM is",linear_svr.score(X_test,y_test))
#使用mean_squared_error模块,并输出评估结果
print("The mean squared error of linear SVM is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_predict)))
#使用mean_absolute_error模块,并输出评估结果
print("The mean absolute error of linear SVM is",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(linear_svr_predict)))
print("===========================")
#使用r2_score模块,并输出评估结果
print("The value of R-squared of Poly SVM is",poly_svr.score(X_test,y_test))
#使用mean_squared_error模块,并输出评估结果
print("The mean squared error of Poly SVM is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_predict)))
#使用mean_absolute_error模块,并输出评估结果
print("The mean absolute error of Poly SVM is",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(poly_svr_predict)))
print("===========================")
#使用r2_score模块,并输出评估结果
print("The value of R-squared of RBF SVM is",rbf_svr.score(X_test,y_test))
#使用mean_squared_error模块,并输出评估结果
print("The mean squared error of RBF SVM is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_predict)))
#使用mean_absolute_error模块,并输出评估结果
print("The mean absolute error of RBF SVM is",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rbf_svr_predict)))






































































