#coding:utf-8
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from PIL import Image

INPUT_NODE=784
OUTPUT_NODE=10
LAYER1_NODE=500
MOVING_AVERAGE_DECAY=0.99
MODEL_SAVE_PATH="./model1/"

def get_weight(shape,regularizer):
    w=tf.Variable(tf.truncated_normal(shape,stddev=0.1))
    if regularizer!=None:
        tf.add_to_collection("losses",tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b=tf.Variable(tf.zeros(shape))
    return b
    
def forward(x,regularizer):
    w1=get_weight([INPUT_NODE,LAYER1_NODE],regularizer)
    b1=get_bias([LAYER1_NODE])
    y1=tf.nn.relu(tf.matmul(x,w1)+b1)
    w2=get_weight([LAYER1_NODE,OUTPUT_NODE],regularizer)
    b2=get_bias([OUTPUT_NODE])
    #由于输出y要经过softmax函数,使其符合概率分布,故输出不经过relu函数
    y=tf.matmul(y1,w2)+b2
    return y
	
def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x=tf.placeholder(tf.float32,[None,INPUT_NODE])
        y=forward(x,None)
        preValue=tf.argmax(y,1) #得到最大概率的预测值
        variable_averages=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
        variables_to_restore=variable_averages.variables_to_restore()
        saver=tf.train.Saver(variables_to_restore)
        with tf.Session() as sess:
            ckpt=tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                preValue=sess.run(preValue,feed_dict={x:testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1
				
def pre_pic(picName):
    img=Image.open(picName)
    reIm=img.resize((28,28),Image.ANTIALIAS)
    plt.show(reIm)	
    im_arr=np.array(reIm.convert("L"))
    threshold=50 #设定合理的阈值
    for i in range(28):
        for j in range(28):
            #得到互补的反色(黑底白色)
            im_arr[i][j]=255-im_arr[i][j]
    nm_arr=im_arr.reshape([1,784])
    nm_arr=nm_arr.astype(np.float32) 
    #0-255 -》0->1
    img_ready=np.multiply(nm_arr,1.0/255.0)
    return img_ready     
    
def application():
    testNum=int(input("input the number of test pictures:"))
    for i in range(testNum):
        testPic=input("the path of test picture:")
        testPicArr=pre_pic(testPic)
        preValue=restore_model(testPicArr)
        print("The prediction number is:",preValue)

application()