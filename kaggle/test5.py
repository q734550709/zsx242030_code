#K近邻(分类)
#需要寻找与待分类的样本在特征空间中距离最近的K个已近标记样本作为参考
#K的选取不同,会获得不同的分类器效果
#K不属于模型通过训练数据学习的参数,因此要在模型初始化的过程中提前确定
#K近邻算法属于无参数模型
#平方级的算法时间复杂度
#使用Iris数据集
from sklearn.datasets import load_iris
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
iris=load_iris()
print(iris.data.shape)
#(150L,4L)
#查看数据的说明
#print(iris.DESCR)
X_train,X_test,y_train,y_test=train_test_split(iris.data,iris.target,test_size=0.25,random_state=33)
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

knc=KNeighborsClassifier()
knc.fit(X_train,y_train)
y_predict=knc.predict(X_test)

print("The Accuracy of k-Nearest Neighbor Classifier is",knc.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=iris.target_names))














































































































































