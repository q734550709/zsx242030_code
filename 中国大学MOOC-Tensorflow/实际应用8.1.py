import tensorflow as tf
with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v=tf.get_variable("v",[0])
        m=tf.get_variable("m",[1])
        assert v.name=="foo/bar/v:0"
        print(v.name)
        print(m.name)

#np.load	np.save:将数组以二进制格式保存到磁盘,扩展名为.npy。
#tf.shape(a)和a.get_shape()
#相同点：都可以得到tensor a 的尺寸
#不同点：tf.shape()中a的数据类型可以是tensor, list, array
#a.get_shape()中a的数据类型只能是tensor, 且返回的是一个元组（ tuple）。
import tensorflow as tf
import numpy as np
x=tf.constant([[1,2,3],[4,5,6]])
y=[[1,2,3],[4,5,6]]
z=np.arange(24).reshape([2,3,4])
sess=tf.Session()
#tf.shape()
x_shape=tf.shape(x)#[2 3]
y_shape=tf.shape(y)#[2 3]
z_shape=tf.shape(z)#[2 3 4]
print(sess.run(x_shape))
print(sess.run(y_shape))
print(sess.run(z_shape))
#a.get_shape()
x_shape=x.get_shape()
print(x_shape)#(2, 3)
# 返回的是 TensorShape([Dimension(2),Dimension(3)])
#不能使用 sess.run()，因为返回的不是tensor或 string,而是元组
x_shape=x.get_shape().as_list()
print(x_shape)#[2, 3]
# 可以使用 as_list()得到具体的尺寸，x_shape=[2 3]
#y_shape=y.get_shape().as_list()
#z_shape=z.get_shape().as_list()
#只有Tensor才有get_shape(),列表和numpy没有get_shape()
#Tensor,列表和numpy都有shape()

#tf.nn.bias_add(乘加和,bias)：把bias加到乘加和上
#np.argsort( 列表) ：对列表从小到大排序。
#os.getcwd():返回当前工作目录。
#os.path.join(path1[,path2[,......]])
#返回值：将多个路径组合后返回。
#注意：第一个绝对路径之前的参数将被忽略。

#import os
#vgg16_path = os.path.join(os.getcwd(),"vgg16.npy")
#当前目录/vgg16.npy，索引到vgg16.npy文件

#np.save：写数组到文件（未压缩二进制形式），文件默认的扩展名是.npy。
#np.save("名.npy"，某数组): 将某数组写入“名.npy ”文件。
#某变量=np.load("名.npy"，encoding=" ").item() ：将“名 .npy ”文件读出给某变量。 
#encoding=" "可以不写 ‘latin1’ 、 ‘ASCII’ 、 ‘bytes’ ，默认为 ’ASCII’ 。

import numpy as np
A = np.arange(15).reshape(3,5)
print(A)
np.save("data3/A.npy",A) 
#如果文件路径末尾没有扩展名.npy，该扩展名会被自动加上。

B=np.load("data3/A.npy")
print(B)


#tf.split(input, num_or_size_splits=num_split, axis=dimension)
#dimension:输入张量的哪一个维度，如果是0就表示对第0维度进行切割。
#num_split：切割的数量，如果是2就表示输入张量被切成2份，每一份是一个列表。

import tensorflow as tf
import numpy as np
A = np.array([[1,2,3],[4,5,6]])
x = tf.split(A, num_or_size_splits=3, axis=1)
with tf.Session() as sess:
	c = sess.run(x)
	for ele in c:
		print(ele)

#tf.concat(values,axis=concat_dim,):沿着某一维度连结tensor ：
t1 = [[1, 2, 3], [4, 5, 6]]
t2 = [[7, 8, 9], [10, 11, 12]]
t3=tf.concat([t1, t2],axis=0)
t4=tf.concat([t1, t2],axis=1)
with tf.Session() as sess:
	print(sess.run(t3))
	print("==========")
	print(sess.run(t4))

#如果想沿着tensor一新轴连结打包,那么可以：
#tf.expand_dims(input, axis=None, name=None, dim=None)
sess = tf.InteractiveSession()
print(sess.run(tf.expand_dims(t1, 0)))

print("#"*40)
sess = tf.InteractiveSession()
print(sess.run(tf.concat([tf.expand_dims(t, 0) for t in [t1,t2]],axis=0)))
print("#"*40)

#tf.concat([tf.expand_dims(t, axis) for t in tensors],axis=)等同于tf.stack(tensor, name="str")
sess = tf.InteractiveSession()
print(sess.run(tf.stack([t1,t2],name='rank')))

#fig = plt.figure(" 图名字 ") ：实例化图对象。
#ax = fig.add_subplot(m,n, k) ： 将画布分割成 m行n 列，图像画在从左到右从上到下的第 k块 。
import matplotlib.pyplot as plt
from numpy import *
#绘图
fig = plt.figure()
ax = fig.add_subplot(3,4,1)
x=np.array([1,2,3])
y=np.array([1,4,9])
ax.plot(x,y)
plt.show()