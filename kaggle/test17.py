#DictVectorizer对使用字典存储的数据进行特征抽取与向量化
from sklearn.feature_extraction import DictVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

count_filter_vec,tfidf_filter_vec=CountVectorizer(analyzer='word',stop_words='english'),TfidfVectorizer(analyzer='word',stop_words='english')

news=fetch_20newsgroups(subset='all')
X_train,X_test,y_train,y_test=train_test_split(news.data,news.target,test_size=0.25,random_state=33)

X_count_filter_train=count_filter_vec.fit_transform(X_train)
X_count_filter_test=count_filter_vec.transform(X_test)


X_tfidf_filter_train=tfidf_filter_vec.fit_transform(X_train)
X_tfidf_filter_test=tfidf_filter_vec.transform(X_test)

mnb_count_filter=MultinomialNB()
mnb_count_filter.fit(X_count_filter_train,y_train)

print("The accuracy of classifing 20newsgroups using Naive Bayes(CountVectorizer by filtering stopwords):",mnb_count_filter.score(X_count_filter_test,y_test))

y_count_filter_predict=mnb_count_filter.predict(X_count_filter_test)

print(classification_report(y_test,y_count_filter_predict,target_names=news.target_names))

print("===============================")

mnb_tfidf_filter=MultinomialNB()
mnb_tfidf_filter.fit(X_tfidf_filter_train,y_train)

print("The accuracy of classifing 20newsgroups using Naive Bayes(TfidfVectorizer by filtering stopwords):",mnb_tfidf_filter.score(X_tfidf_filter_test,y_test))

y_tfidf_filter_predict=mnb_tfidf_filter.predict(X_tfidf_filter_test)

print(classification_report(y_test,y_tfidf_filter_predict,target_names=news.target_names))


















































