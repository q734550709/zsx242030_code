#集成模型
from sklearn.cross_validation import train_test_split
from sklearn.datasets import load_boston
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor,GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import numpy as np

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

#RandomForestRegressor
rfr=RandomForestRegressor()
rfr.fit(X_train,y_train)
rfr_y_predict=rfr.predict(X_test)

#ExtraTreesRegressor
etr=ExtraTreesRegressor()
etr.fit(X_train,y_train)
etr_y_predict=etr.predict(X_test)

#GradientBoostingRegressor
gbr=GradientBoostingRegressor()
gbr.fit(X_train,y_train)
gbr_y_predict=gbr.predict(X_test)

print("===========================")

print("The value of R-squared of RandomForestRegressor is",rfr.score(X_test,y_test))
print("The mean squared error of RandomForestRegressor is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rfr_y_predict)))
print("The mean absolute error of RandomForestRegressor is",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(rfr_y_predict)))

print("===========================")

print("The value of R-squared of ExtraTreesRegressor is",etr.score(X_test,y_test))
print("The mean squared error of ExtraTreesRegressor is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(etr_y_predict)))
print("The mean absolute error of ExtraTreesRegressor is",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(etr_y_predict)))

print("==================================================")

#利用训练好的极端回归森林模型,输出每种特征对预测目标的贡献度
print(np.sort(list(zip(etr.feature_importances_,boston.feature_names)),axis=0))

print("===========================")

print("The value of R-squared of GradientBoostingRegressor is",gbr.score(X_test,y_test))
print("The mean squared error of GradientBoostingRegressor is",mean_squared_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(gbr_y_predict)))
print("The mean absolute error of GradientBoostingRegressor is",mean_absolute_error(ss_y.inverse_transform(y_test),ss_y.inverse_transform(gbr_y_predict)))

print("==================================================")


































































