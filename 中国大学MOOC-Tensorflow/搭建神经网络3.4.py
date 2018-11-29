#coding=utf-8
#0导入模块,生成模拟数据集
import tensorflow as tf 
import numpy as np
BATCH_SIZE=8
seed=23455
#基于seed产生随机数
rng=np.random.RandomState(seed)
#输入数据集
X=rng.rand(32,2) #(32,2)
#对于X的每一行
#输入数据集的标签
Y=[[int(X0+X1<1)] for (X0,X1) in X]
print("X:")
print(X)
print("Y:")
print(Y)
#===============================================================
#1.定义神经网络的输入,参数和输出,定义前向传播过程
x=tf.placeholder(tf.float32,shape=(None,2))
y_=tf.placeholder(tf.float32,shape=(None,1))

W1=tf.Variable(tf.random_normal([2,3],stddev=1,seed=1))
W2=tf.Variable(tf.random_normal([3,1],stddev=1,seed=1))

a=tf.matmul(x,W1)
y=tf.matmul(a,W2)

#2.定义损失函数以及反向传播方法
loss=tf.reduce_mean(tf.square(y-y_))

#这里使用梯度下降
train_step=tf.train.GradientDescentOptimizer(0.001).minimize(loss)

#3.生成会话,训练STEP轮
with tf.Session() as sess:
	init_op=tf.global_variables_initializer()
	sess.run(init_op)
	print("W1:")
	print(sess.run(W1))
	print("W2:")
	print(sess.run(W2))
	STEPS=3000
	for i in range(STEPS):
		start=(i*BATCH_SIZE)%32
		end=start+BATCH_SIZE
		sess.run(train_step,feed_dict={x:X[start:end],y_:Y[start:end]})
		if i%500==0:
			total_loss=sess.run(loss,feed_dict={x:X,y_:Y})
			print("After %d training step(s),loss on all data is %g"%(i,total_loss))
	print("\n")
	print("w1:")
	print(sess.run(W1))
	print("w2:")
	print(sess.run(W2))