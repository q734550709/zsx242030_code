import tensorflow as tf
import numpy as np
greeting=tf.constant("Hello Google Tensorflow")
matrix1=tf.constant([[3.,3.]])
matrix2=tf.constant([[2.],[2.]])
product=tf.matmul(matrix1,matrix2)
linear=tf.add(product,tf.constant(2.0))
sess=tf.Session()
result=sess.run(greeting)
print(result)
sess.close()
with tf.Session() as sess:
    result=sess.run(linear)
    print(result)
