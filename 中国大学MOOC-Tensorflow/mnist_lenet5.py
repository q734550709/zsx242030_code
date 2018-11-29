import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import os
import time

#mnist_lenet5_forward.py

#设定神经网络的超参数
IMAGE_SIZE=28 #图片的尺寸
NUM_CHANNELS=1 #通道数
CONV1_SIZE=5 #卷积核的大小
CONV1_KERNEL_NUM=32 #卷积核个数
CONV2_SIZE=5
CONV2_KERNEL_NUM=64
FC_SIZE=512 #全连接层第一层为512个神经元
OUTPUT_NODE=10 #全连接层第二层为10个神经元,实现10分类输出

#权重W生成函数
#regularizer正则化项的权重
def get_weight(shape,regularizer):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer!=None:
        #为权重加入L2正则化,通过限制权重的大小使模型不会随意拟合训练数据中机噪音
        tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

#偏置b生成函数
def get_bias(shape):
    b=tf.Variable(tf.zeros(shape))
    return b

#卷积层计算函数
#x一个输入batch
#strides表示卷积核在不同维度上的移动步长为1,第一维和第四一定是1,这是因为卷积层的步长只对矩阵的和宽有效
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")

#eg:
#tf.nn.conv2d(输入描述[batch,行分辨率,列分辨率,通道数],
#            卷积核描述[行分辨率,列分辨率,通道数,卷积核个数],
#            核滑动步长[1,行步长,列步长,1],
#            填充模式padding)
#tf.nn.conv2d(x=[100,28,28,1],x=[5,5,1,6],strides=[1,1,1,1],padding="SAME")
#本例表示卷积输入x为28*28*1,一个batch_size为100,卷积核大小为5*5,卷积核个数为6,垂直方#向步长为1,水平方向步长为1,填充方式为全零.
	
#最大池化层计算函数
#ksize表示池化过滤器的边长为2,strides表示过滤器移动步长是2,'SAME' 提供使用全 提供使用全0填
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#eg.
#tf.nn.max_pool(输入描述[batch,行分辨率,列分辨率,通道数],
#              池化核描述[1,行分辨率,列分辨率,1],
#              池化核滑动步长[1,行步长,列步长,1],
#              填充模式adding)
#tf.nn.max_pool(x=[100,28,28,1],ksize=[1,2,2,1],strides=[1,2,2,1],padding=["SAME"])	
#本例表示卷积输入x为28*28*1,一个batch_size为100,池化核大小用ksize,第一维和四都为1,池化核大小为2*2,垂直方向步长为1,水平方向步长为1.

#train:用于区分训练过程True,测试过程False
def forward(x,train,regularizer):
    #实现第一层卷积
    #(5,5,1,32)
    conv1_w=get_weight([CONV1_SIZE,CONV1_SIZE,NUM_CHANNELS,CONV1_KERNEL_NUM],regularizer)
    #(32)
    conv1_b=get_bias([CONV1_KERNEL_NUM])
    #x*w(28x28x32)
    conv1=conv2d(x,conv1_w)
    #x*w+b并且进入激活函数relu,可以进行快速的收敛
    relu1=tf.nn.relu(tf.nn.bias_add(conv1,conv1_b))
    #最大池化(14x14x32)
    pool1=max_pool_2x2(relu1)
	#实现第二层卷积
    #(5,5,32,64)
    #该层每个卷积核的通道数要与上一致
    conv2_w=get_weight([CONV2_SIZE,CONV2_SIZE,CONV1_KERNEL_NUM,CONV2_KERNEL_NUM],regularizer)
    #(64)
    conv2_b=get_bias([CONV2_KERNEL_NUM])
    #计算(14x14x64)
    conv2=conv2d(pool1,conv2_w)
    #进入激活函数
    relu2=tf.nn.relu(tf.nn.bias_add(conv2,conv2_b))
    #最大池化(7x7x64)
    pool2=max_pool_2x2(relu2)
	#将第二层池化层的输出pool2矩阵转化为全连接层的输入格式即向量形
    #get_shape()函数得到pool2输出矩阵的维度,并存入list中.
    #其中,shape[0]为一个batch值shape[1],shape[2],shape[3]为长宽和深度
    #(7,7,64)
    pool_shape=pool2.get_shape().as_list()
    #从list中依次取出矩阵的长宽及深度,并求三者乘积得到矩阵被拉长后的长度 。
    nodes=pool_shape[1]*pool_shape[2]*pool_shape[3]
    #将pool2转换为一个batch的向量再传入后续全连接.
    #(1,7x7x64)
    reshaped=tf.reshape(pool2,[pool_shape[0],nodes])
    #实现第三层全连接层
    #(3136,512)
    fc1_w=get_weight([nodes,FC_SIZE],regularizer)
    #(512)
    fc1_b=get_bias([FC_SIZE])
    #计算(1,3136)*(3136,512)=(1,512)
    fc1=tf.nn.relu(tf.matmul(reshaped,fc1_w)+fc1_b)
    #如果是训练阶段，则对该层输出使用dropout,也就是随机的将该层输出中一半神经元置为无效
    #是为了避免过拟合而设置的一般只在全连接层中使用.如果是训练阶段，则对该层输出使用 dropout,也就是随机的将该层输出中一半神经元置为无效.
    if train:
        fc1=tf.nn.dropout(fc1,0.5)
    #实现第四层全连接层的前向传播过程：
    #(512,10)
    fc2_w=get_weight([FC_SIZE,OUTPUT_NODE],regularizer)
    #(10)
    fc2_b=get_bias([OUTPUT_NODE])
    #(1,512)*(512,10)=(1,10)
    y=tf.matmul(fc1,fc2_w)+fc2_b	
    return y

#mnist_lenet5_backward.py

#定义训练过程中的超参数
BATCH_SIZE=100 #一个batch的数量
LEARNING_RATE_BASE=0.005 #初始学习率
LEARNING_RATE_DECAY=0.99 #学习衰减率
REGULARIZER=0.0001 #正则化项的权重
STEPS=50000 #最大迭代次数
MOVING_AVERAGE_DECAY=0.99 #滑动平均衰减率
MODEL_SAVE_PATH="./model/" #保存模型的路径
MODEL_NAME="mnist_model" #模型命名

def backward(mnist):
    x=tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])
    y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE])
    #调用前向传播网络得到维度为10的tensor
    y=forward(x,True,REGULARIZER)
    #求含有正则化的损失值
    #声明一个全局计数器,并初始化为0
    global_step=tf.Variable(0,trainable=False)
    #先是对网络最后一层的输出y做softmax
    #通常是求取输出属于某一类的概率，其实就是一个num_classe大小的向量
    #再将此向量和实际标签值做交叉熵，需要说明的是该函数返回一个向量 
    ce=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cem=tf.reduce_mean(ce)
    loss=cem+tf.add_n(tf.get_collection("losses"))
    #实现指数衰减学习率
    #实现指数级的减小学习率,可以让模型在训
	#练前期快速接近较优解训练后期不会有太大波动
    #计算公式:decayed_learning_rate=learning_rate*decay_rate^(global_step/decay_step)
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,mnist.train.num_examples/BATCH_SIZE,LEARNING_RATE_DECAY,staircase=True)
    #传入学习率，构造一个实现梯度下降算法的优化器再通过使用minimize
    #更新存储要训练的变量的列表来减小loss
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step=global_step)
    #实现滑动平均模型
    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,global_step)
    ema_op=ema.apply(tf.trainable_variables())
    #将train_step和ema_op两个训练操作绑定到train_op上
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op(name="train")
    #实例化一个保存和恢复变量的saver
    saver=tf.train.Saver()
    with tf.Session() as sess:
        init_op=tf.global_variables_initializer()
        sess.run(init_op)
        #通过checkpoint文件定位到最新保存的模型
        ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess,ckpt.model_checkpoint_path)#加载最新的模型
        for i in range(STEPS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE) #读取一个batch的数据
            #将输入数据xs转换成与网络输入相同形状的矩阵(100,28,28,1)
            reshaped_xs=np.reshape(xs,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])
            #喂入训练图像和标签,开始训练
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:reshaped_xs,y_:ys})
            #每迭代 100100100次打印loss信息，并保存最新的模型 
            if i%100==0:
                print("After %d training step(s),loss on training batch is %g."%(step,loss_value))			
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step=global_step)

#mnist_lenet5_test.py				

TEST_INTERVAL_SECS=5 #暂停时间

def test(mnist):
    #创建一个默认图，在该中执行以下操作
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[mnist.test.num_examples,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS])
        y_=tf.placeholder(tf.float32,[None,OUTPUT_NODE])
        y=forward(x,False,None)
        ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        ema_restore=ema.variables_to_restore()
        saver=tf.train.Saver(ema_restore)
        #判断预测值和实际值是否相等
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        #求平均得到准确率
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    while True:
        with tf.Session() as sess: 
            ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                #根据读入的模型名字切分出该模型是属于迭代了多少次保存的
                global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                reshaped_x=np.reshape(mnist.test.images,(mnist.test.num_examples,IMAGE_SIZE,IMAGE_SIZE,NUM_CHANNELS))
                accuracy_score=sess.run(accuracy,feed_dict={x:reshaped_x,y_:mnist.test.labels}) #计算出测试集上准确率
                print("After %d training step(s),test accuracy=%g"%(global_step,accuracy_score))			
            else:
                print("No checkpoint file found")
                return 					
        time.sleep(TEST_INTERVAL_SECS)#每隔5秒寻找一次是否有最新的模型

def main():
    mnist=input_data.read_data_sets("./data/",one_hot=True)
    test(mnist)

if __name__=="__main__":
    main()	