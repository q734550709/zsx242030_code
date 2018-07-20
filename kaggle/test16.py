#DictVectorizer对使用字典存储的数据进行特征抽取与向量化
from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

measurements=[{'city':'Dubai','temperature':33.},{'city':'Londo','temperature':12.},{'city':'San Fransisco','temperature':18.}]
vec=DictVectorizer()
print(vec.fit_transform(measurements).toarray())
print(vec.get_feature_names())

#文本特征表示法为词袋法
#词表：不重复的词汇集合
#两种：CountVectorizer和TfidfVectorizer
#训练文本较多的时候使用TfidfVectorizer
#停用词：每条文本中都出现的常用词（都以黑名单的方式进行过滤）

news=fetch_20newsgroups(subset='all')
X_train,X_test,y_train,y_test=train_test_split(news.data,news.target,test_size=0.25,random_state=33)

#CountVectorizer默认配置不去除英文停用词
count_vec=CountVectorizer()
X_count_train=count_vec.fit_transform(X_train)
X_count_test=count_vec.transform(X_test)

mnb_count=MultinomialNB()
mnb_count.fit(X_count_train,y_train)

print("The accuracy of classifing 20newsgroups using Naive Bayes(CountVectorizer without filtering stopwords):",mnb_count.score(X_count_test,y_test))

y_count_predict=mnb_count.predict(X_count_test)
print(classification_report(y_test,y_count_predict,target_names=news.target_names))

print("===============================")

tfidf_vec=TfidfVectorizer()

X_tfidf_train=tfidf_vec.fit_transform(X_train)
X_tfidf_test=tfidf_vec.transform(X_test)

mnb_tfidf=MultinomialNB()
mnb_tfidf.fit(X_tfidf_train,y_train)

print("The accuracy of classifing 20newsgroups using Naive Bayes(TfidfVectorizer without filtering stopwords):",mnb_tfidf.score(X_tfidf_test,y_test))

y_tfidf_predict=mnb_tfidf.predict(X_tfidf_test)
print(classification_report(y_test,y_tfidf_predict,target_names=news.target_names))

#过滤掉停用词可以提高性能

















































