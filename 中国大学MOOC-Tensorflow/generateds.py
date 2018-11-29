#1>生成数据集
import tensorflow as tf
import numpy as np
def to_generateds():
    seed=2
    rdm=np.random.RandomState(seed) 
    #1.准备数据
    X=rdm.randn(300,2)
    Y_=[int(x0*x0+x1*x1<2) for (x0,x1) in X]
    Y_c=[['red' if y else 'blue'] for y in Y_]
    X=np.vstack(X).reshape(-1,2)
    Y_=np.vstack(Y_).reshape(-1,1)
    return X,Y_,Y_c

X,Y_,Y_c=to_generateds()
print(X.shape)
print(Y_.shape)
print(len(Y_c))
print(Y_c)#[[],[],[],...]