import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
#为True时,表示以独热码形式存取数据集
mnist=input_data.read_data_sets("./data/",one_hot=True)
#将mnist数据集分为训练集train和验证集validation和测试集test存放
print(mnist)
#训练集train的数量
print("train data size:",mnist.train.num_examples)
#验证集validation的数量
print("validation data size:",mnist.validation.num_examples)
#测试集test的数量
print("test data size:",mnist.test.num_examples)
#使用train.labels函数返回mnist数据集标签
print(mnist.train.labels[0])
#使用train.images函数返回mnist数据集图片像素
print(mnist.train.images[0])
#使用mnist.train.next_batch()函数将数据输入神经网络
BATCH_SIZE=200
#随机从数据集和标签中取出20条数据
xs,ys=mnist.train.next_batch(BATCH_SIZE)
print("xs shape:",xs.shape) #(200,784)
print("ys shape:",ys.shape) #(200,10)

#(1)tf.get_collection("")函数表示从collection集合中取出全部变量生一个列表
#(2)tf.add()函数表示将参数列表中对应元素相加
x=tf.constant([[1,2],[1,2]])
y=tf.constant([[1,1],[1,2]])
z=tf.add(x,y)
print(z) #并不直接计算
#(3)tf.cast(x,dtype)函数表示将参数x转换为指定的数据类型
A=tf.convert_to_tensor(np.array([[1,1,2,4],[3,4,8,5]]))
print(A.dtype)
B=tf.cast(A,tf.float32)
print(B.dtype)
#(4)tf.equal()函数表示对比两个矩阵或者向量的元素.若两个元素相等,则返回True,若两个元素不相等,则返回False
A=[[1,3,4,5,6]]
B=[[1,3,4,3,2]]
with tf.Session() as sess:
    print(sess.run(z))
    print(sess.run(tf.equal(A,B)))
#(5)tf.reduce_mean(x,axis)函数表示求取矩阵或张量指定维度的平均值。若不指定第二个参数,则在所有元素中取平均数,若指定第二个参数为0,则在第一维元素上取平均值,即每一列求平均值;若指定第二个参数为1,则在第二维元素上取平均值,即每一行求平均值。
x=[[1.,1.],[2.,2.]]
print(tf.reduce_mean(x))
print(tf.reduce_mean(x,0))
print(tf.reduce_mean(x,1))
#(6)tf.argmax(x,axis)函数表示返回指定维度axis下,参数x中最大值索引号。
#tf.argmax([1,0,0],1)
#(7)os.path.join()函数表示把参数字符串按照路径命名规则拼接
import os
print(os.path.join("/hello/","good/boy/","doiido"))
#(8)字符串.split()函数表示按照指定"拆分符"对字符串拆分,返回拆分列表
print("/model/mnist_model-1001".split("/")[-1].split("-")[-1])
