#K近邻(回归)
#配置不同的K近邻模型来比较回归性能的差异
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
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

#预测的方式为平均回归
uni_knr=KNeighborsRegressor(weights='uniform')
uni_knr.fit(X_train,y_train)
uni_knr_y_predict=uni_knr.predict(X_test)

#预测的方式为根据距离加权回归
dis_knr=KNeighborsRegressor(weights='distance')
dis_knr.fit(X_train,y_train)
dis_knr_y_predict=dis_knr.predict(X_test)

print("===========================")

#使用r2_score模块,并输出评估结果
print("The value of R-squared of uniform-weighted KNeighborRegression is",uni_knr.score(X_test,y_test))
#使用mean_squared_error模块,并输出评估结果
print("The mean squared error of uniform-weighted KNeighborRegression is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_knr_y_predict)))
#使用mean_absolute_error模块,并输出评估结果
print("The mean absolute error of uniform-weighted KNeighborRegression is",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(uni_knr_y_predict)))


print("===========================")
#使用r2_score模块,并输出评估结果
print("The value of R-squared of distance-weighted KNeighborRegression is",dis_knr.score(X_test,y_test))
#使用mean_squared_error模块,并输出评估结果
print("The mean squared error of distance-weighted is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dis_knr_y_predict)))
#使用mean_absolute_error模块,并输出评估结果
print("The mean absolute error of distance-weighted is",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(dis_knr_y_predict)))

#采用K近邻加权平均的回归策略可以获得较高的模型性能







































































