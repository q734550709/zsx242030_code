import tensorflow as tf
import numpy as np

#Lenet进行微调的结构如下图所示：
#############  28x28x1
###      conv1    ↓     5x5x1x32  全0 步长1
#############  28x28x32 
###      pooling  ↓     2x2       全0 步长2
#############  14x14x32  
###      conv2    ↓     5x5x32x64 全0 步长1
#############  14x14x64
###      pooling  ↓     2x2       全0 步长2
#############  7x7x64
###      拉直    ↓      [1,7x7x64]
#############  全连接

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
        tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

#偏置b生成函数
def get_bias(shape):
    b=tf.Variable(tf.zeros(shape))
    return b

#卷积层计算函数
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding="SAME")

#eg:
#tf.nn.conv2d(输入描述[batch,行分辨率,列分辨率,通道数],
#            卷积核描述[行分辨率,列分辨率,通道数,卷积核个数],
#            核滑动步长[1,行步长,列步长,1],
#            填充模式padding)
#tf.nn.conv2d(x=[100,28,28,1],w=[5,5,1,6],strides=[1,1,1,1],padding="SAME")
#本例表示卷积输入x为28*28*1,一个batch_size为100,卷积核大小为5*5,卷积核个数为6,垂直方#向步长为1,
#水平方向步长为1,填充方式为全零.
	
#最大池化层计算函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

#eg.
#tf.nn.max_pool(输入描述[batch,行分辨率,列分辨率,通道数],
#              池化核描述[1,行分辨率,列分辨率,1],
#              池化核滑动步长[1,行步长,列步长,1],
#              填充模式adding)
#tf.nn.max_pool(x=[100,28,28,1],ksize=[1,2,2,1],strides=[1,2,2,1],padding=["SAME"])	
#本例表示卷积输入x为28*28*1,一个batch_size为100,池化核大小用ksize,第一维和四都为1,池化核大小为2*2,垂直方向步长为1,水平方向步长为1.

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
    #是为了避免过拟合而设置的一般只在全连接层中使用.
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