#=========================================决策树=================================
#常用的度量方式为信息熵（Information Gain）和基尼不纯性(Gini Impurity)
#使用的是泰坦尼克号沉船事故
import pandas as pd
from sklearn.cross_validation import train_test_split
#使用特征转换器
from sklearn.feature_extraction import DictVectorizer
#导入决策树
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')
print(titanic.head())
print(titanic.info()) 

#由于有数据的丢失,所以需要做数据处理
X=titanic[['pclass','age','sex']]
y=titanic['survived']
print(X.shape) #(1313,3)
print(y.shape) #(1313,1)
print(X.info()) 

#使用中位数或者是平均数来填充
X['age'].fillna(X['age'].mean(),inplace=True)

print(X.info())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

vec=DictVectorizer(sparse=False)

#转换特征后,我们发现凡是类别型的特征都单独剥离出来,独成一列特征,数值型的则保持不变
X_train=vec.fit_transform(X_train.to_dict(orient='record'))
print(vec.feature_names_)

#同样需要对测试数据进行转换
X_test=vec.fit_transform(X_test.to_dict(orient='record'))

dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)

y_predict=dtc.predict(X_test)

print(dtc.score(X_test,y_test))

print(classification_report(y_predict,y_test,target_names=['died','survived']))






























































