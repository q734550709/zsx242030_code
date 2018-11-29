#喂入神经网络，构建一个层的拟合预测酸奶日销量函数。
#coding:utf-8
#预测多或预测少的影响一样
import  numpy as np
import tensorflow as tf
BATCH_SIZE=8
SEED=23455
rdm=np.random.RandomState(SEED)
#输入数据集
X=rdm.rand(32,2)
#实际的输出
Y_=[[x1+x2+(rdm.rand()/10.0-0.05)] for (x1,x2) in X]

x=tf.placeholder(tf.float32,shape=(None,2))

w1=tf.Variable(tf.random_normal([2,1],stddev=1,seed=1))

y_=tf.placeholder(tf.float32,shape=(None,1))

y=tf.matmul(x,w1)

loss_me=tf.reduce_mean(tf.square(y_-y))

train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss_me)

with tf.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    STEPS=20000
    for i in range(STEPS):
        start=(i*BATCH_SIZE)%32
        end=(i*BATCH_SIZE)%32+BATCH_SIZE
        sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
        if i%500==0:
            print("After %d training steps,wl is:"%(i))
            print(sess.run(w1),"\n")
    print("Final wl is:\n",sess.run(w1))