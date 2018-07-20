#提升(Boosting)分类器隶属于集成学习模型
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import DictVectorizer
#采用默认配置的XGBoost模型相同的测试进行预测
from xgboost import XGBClassifier
#随机森林
from sklearn.ensemble import RandomForestClassifier

titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

X=titanic[['pclass','age','sex']]
y=titanic['survived']

X['age'].fillna(X['age'].mean(),inplace=True)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

vec=DictVectorizer(sparse=False)

X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.fit_transform(X_test.to_dict(orient='record'))

rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred=rfc.predict(X_test)

print("The accuracy of decision tree is,",rfc.score(X_test,y_test))

xgbc=XGBClassifier()
xgbc.fit(X_train,y_train)
print("The accuracy of decision tree is,",xgbc.score(X_test,y_test))