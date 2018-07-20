#朴素贝叶斯
#朴素贝叶斯分类器的构造基础是贝叶斯理论
#各个维度上的特征被分类的条件概率之间是相互独立的
#采用概率模型来表述：
#则定义：x=<x1,x2,x3,......,xn>为某一n维特征向量
#y属于{c1,c2,c3,......,ck}为该特征向量x所有k种可能的类别，
#记P(y=ci|x)为特征向量x属于类别ci的概率。
#贝叶斯原理：
#P(y|x)=P(x|y)P(y)/P(x)
#我们的目标是寻找所有Y属于{c1,c2,c3,.....ck}中P(y|x)最大的,即argmaxP(y|x)
#P(X1,X2,...,Xn|y)=P(X1|y)p(X2|X1,y)P(X3|X1,X2,y)...P(Xn|X1,X2,X3,...,X(n-1),y)
#P(Xn=1|y=Ck)=P(Xn=1,y=Ck)/P(y=Ck)=#(Xn=1,y=Ck)/#(y=Ck)
#==========================================文本分类===================================
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
#导入用于文本特征向量转换模块
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
news=fetch_20newsgroups(subset='all')
print(len(news.data))
print(news.data[0])

print("===================================")
#这些文本数据既没有被设定特征,也没有数字化的量度,需要对数据做进一步的处理。
X_train,X_test,y_train,y_test=train_test_split(news.data,news.target,test_size=0.25,random_state=33)
vec=CountVectorizer()
X_train=vec.fit_transform(X_train)
X_test=vec.transform(X_test)

mnb=MultinomialNB()
mnb.fit(X_train,y_train)
y_predict=mnb.predict(X_test)

print("Accuracy of Linear SVC Classifier:",mnb.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=news.target_names))

#朴素贝叶斯被广泛应用于海量互联网文本分类任务





















































