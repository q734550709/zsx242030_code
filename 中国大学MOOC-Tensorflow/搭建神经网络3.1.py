import numpy as np
import tensorflow as tf

a=tf.constant([1.0,2.0])
b=tf.constant([3.0,4.0])
result=a+b
print(result)

x=tf.constant([[1.0,2.0]])
w=tf.constant([[3.0],[4.0]])
y=tf.matmul(x,w)
print(y)

with tf.Session() as sess:
    print(sess.run(result))
    print(sess.run(y))
w1=tf.Variable(tf.random_normal([2,3],stddev=2,mean=0,seed=1))
print(w1)

w2=tf.Variable(tf.truncated_normal([2,3],stddev=2,mean=0,seed=1))
print(w2)

w3=tf.Variable(tf.random_uniform(shape=[7,],minval=0,maxval=1,dtype=tf.int32,seed=1))
print(w3)

w4=tf.zeros([3,2],tf.int32)
print(w4)

w5=tf.ones([3,2],tf.int32)
print(w5)

w6=tf.fill([3,2],6)
print(w6)

w7=tf.constant([3,2,1])
print(w7)