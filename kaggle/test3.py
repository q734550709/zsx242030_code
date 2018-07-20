#分类中的SVM(Supported Vector Classifier)
#支持向量
#支持向量机在手写体数字图片的分类任务上展现了良好的性能
#从通过数据加载器获得手写体数字的数码图像数据并存储在digits变量中
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
#标准化数据的模块
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
digits=load_digits()
#检视数据规模和特征维度
print(digits.data.shape)
#(1797L,64L)
#每附图片是由8*8=64的像素矩阵表示
X_train,X_test,y_train,y_test=train_test_split(digits.data,digits.target,test_size=0.25,random_state=33)
#分别监视训练与测试数据规模
print(y_train.shape) #(1347,)
print(y_test.shape)  #(450,)

#训练基于SVM的模型
#数据标准化
ss=StandardScaler()
X_train=ss.fit_transform(X_train)
X_test=ss.transform(X_test)

lsvc=LinearSVC()
lsvc.fit(X_train,y_train)

y_predict=lsvc.predict(X_test)

print("Accuracy of Linear SVC Classifier:",lsvc.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=digits.target_names.astype(str)))






















































































