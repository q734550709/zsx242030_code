#搭建神经网络
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
	
#定义神经网络的输入,参数和输出,定义前向传播过程
def get_weight(shape,regularizer):
    w=tf.Variable(tf.random_normal(shape),dtype=tf.float32)
    tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b=tf.Variable(tf.constant(0.01,shape=shape))
    return b
	
def to_forward(x,regularizer):
    w1=get_weight([2,11],regularizer)
    b1=get_bias([11])
    y1=tf.nn.relu(tf.matmul(x,w1)+b1)
    w2=get_weight([11,1],regularizer)
    b2=get_bias([1])
    y=tf.matmul(y1,w2)+b2 #输出层不过激活函数
	
STEPS=40000
BATCH_SIZE=30
LEARNING_RATE_BASE=0.001
LEARNING_RATE_DESCY=0.999
REGULARIZER=0.01

def backward():
    x=tf.placeholder(tf.float32,shape=(None,2))
    y_=tf.placeholder(tf.float32,shape=(None,1))
    X,Y_,Y_c=to_generateds()
    y=to_forward(x,REGULARIZER)
    global_step=tf.Variable(0,trainable=False)
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,300/BATCH_SIZE,LEARNING_RATE_DESCY,staircase=True)
    #定义损失函数
    loss_mse=tf.reduce_mean(tf.square(y-y_))
    loss_total=loss_mse+tf.add_n(tf.get_collection("losses"))
    #定义反向传播方法：包含正则化
    train_step=tf.train.AdamOptimizer(learning_rate).minimize(loss_total)
    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        STEPS=40000
        for i in range(STEPS):
            start=(i*BATCH_SIZE)%300
            end=start+BATCH_SIZE
            sess.run(train_step,feed_dict={x:X[start:end],y_:Y_[start:end]})
            if i%2000==0:
                loss_mse_v=sess.run(loss_mse,feed_dict={x:X,y_:Y_})
                print("After %d steps,loss is:%f"%(i,loss_mse_v))
        xx,yy=np.mgrid[-3:3:0.1,-3:3:0.1] 
        grid=np.c_[xx.ravel(),yy.ravel()]
        probs=sess.run(y,feed_dict={x:grid})
        probs=probs.reshape(xx.shape)
        print("W1:\n",sess.run(w1))
        print("b1:\n",sess.run(b1))
        print("W1:\n",sess.run(w2))
    plt.scatter(X[:,0],X[:,1],c=np.squeeze(Y_c))
    plt.contour(xx,yy,probs,levels=[.5])
    plt.show() 
if __name__=="__main__":
    backward()