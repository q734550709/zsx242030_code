#特征降维（PCA）
#实用PVA降维和SVM
#=======================显示手写数字图片经PCA压缩后的二位空间的分布======================
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.svm imp
ort LinearSVC
from sklearn.metrics import classification_report

digits_train=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra",header=None)
digits_test=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes",header=None)

X_train=digits_train[np.arange(64)]
y_train=digits_train[64]

X_test=digits_test[np.arange(64)]
y_test=digits_test[64]

svc=LinearSVC()
svc.fit(X_train,y_train)
y_predict=svc.predict(X_test)

#压缩到20个维度
estimator=PCA(n_components=20)
pca_X_train=estimator.fit_transform(X_train)
pca_X_test=estimator.transform(X_test)

pca_svc=LinearSVC()
pca_svc.fit(pca_X_train,y_train)
pca_y_predict=pca_svc.predict(pca_X_test)

print(svc.score(X_test,y_test))
print(classification_report(y_test,y_predict,target_names=np.arange(10).astype(str)))

print(pca_svc.score(pca_X_test,y_test))
print(classification_report(y_test,pca_y_predict,target_names=np.arange(10).astype(str)))



















































































