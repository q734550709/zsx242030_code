#模型检验方法分为留一验证和交叉验证
#留一验证：从任务提供的数据中,随机采样一定比例作为训练集,剩下的留作验证
#交叉验证：从事了多次留一验证的过程。
#每次检验使用的数据集之间是互斥的,并且要保证每一条可用数据都被模型验证过。
#5折交叉验证:每次选取一组作为验证集,其余四组作为训练集

#模型的超参数:(Hyperparameters),如K,不同的核函数等
#超参数搜索方法：网格搜索

#使用单线程对文本分类的朴素贝叶斯模型的超参数组合执行网格搜索
from sklearn.datasets import fetch_20newsgroups
from sklearn.cross_validation import train_test_split
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
news=fetch_20newsgroups(subset='all')
X_train,X_test,y_train,y_test=train_test_split(news.data[:3000],news.target[:3000],test_size=0.25,random_state=33)
#使用Pipeline简化系统搭建流程,将文本抽取与分类器模型串联起来
clf=Pipeline([('vect',TfidfVectorizer(stop_words='english',analyzer='word')),('svc',SVC())])
parameters={'svc_gamma':np.logspace(-2,1,4),'svc_C':np.logspace(-1,1,3)}
gs=GridSearchCV(clf,parameters,verbose=2,refit=True,cv=3)
gs.fit(X_train,y_train)
print(gs.best_params,gs.best_score_)
print(gs.score(X_test,y_test))

#并行搜索
gs=GridSearchCV(clf,parameters,verbose=2,refit=True,cv=3，n_jobs=-1)

























































