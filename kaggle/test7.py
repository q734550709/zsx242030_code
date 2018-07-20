#===================================集成学习===========================================
#集成分类模型便是综合考量多个分类器的预测结果,从而做出决策。
#大体上分为两种：
#一种是利用相同的训练数量同时搭建多个独立的分类模型,然后通过投票的方式,以少数服从多数的原则做出最终的分类决策。代表性的有随机森林分类器。
#另一种则是按照一定次序搭建多个分类模型。
#这些模型之间彼此存在依赖关系。代表性的为梯度提升决策树(Gradient Tree Boosting)
#单一决策树和随机森林分类以及梯度上升决策树的比较
import pandas as pd
#使用特征转换器
from sklearn.feature_extraction import DictVectorizer
from sklearn.cross_validation import train_test_split
#决策器
from sklearn.tree import DecisionTreeClassifier
#随机森林
from sklearn.ensemble import RandomForestClassifier
#梯度提升决策树
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
titanic=pd.read_csv('http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt')

X=titanic[['pclass','age','sex']]
y=titanic['survived']

X['age'].fillna(X['age'].mean(),inplace=True)

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=33)

vec=DictVectorizer(sparse=False)

X_train=vec.fit_transform(X_train.to_dict(orient='record'))
X_test=vec.fit_transform(X_test.to_dict(orient='record'))

dtc=DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc_y_pred=dtc.predict(X_test)

rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
rfc_y_pred=rfc.predict(X_test)

gbc=GradientBoostingClassifier()
gbc.fit(X_train,y_train)
gbc_y_pred=gbc.predict(X_test)

print("The accuracy of decision tree is,",dtc.score(X_test,y_test))
print(classification_report(dtc_y_pred,y_test))

print("The accuracy of random forest classifier is,",rfc.score(X_test,y_test))
print(classification_report(rfc_y_pred,y_test,target_names=['died','survived']))

print("The accuracy of gradient tree boosting is,",gbc.score(X_test,y_test))
print(classification_report(gbc_y_pred,y_test))

#单一决策树<随机森林分类<梯度上升决策树
#经常使用随机森林分类模型作为基线系统






















































