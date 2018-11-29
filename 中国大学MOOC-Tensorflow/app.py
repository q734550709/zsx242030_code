#coding:utf-8
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import vgg16
import utils
#from Nclasses import labels
img_path=input("Input the path and image name:")
img_ready=utils.load_image(img_path) #调用load_image()函数,对测试的图像做一些预处理操作
print("img_ready shape",tf.Session().run(tf.shape(img_ready)))
#定义一个figure画图窗口,并指定窗口的名字,也可以设置窗口的大小
fig=plt.figure(u"Top-5 预测结果")
with tf.Session() as sess:
    #定义一个维度为[1,244,244,3],类型为float32的tensor的占位符
    x=tf.placeholder(tf.float32,[1,244,244,3])
    vgg=vgg16.Vgg16() #类Vgg16实例化出vgg 
    #调用类的成员方法forward(),并传入待测试的图像,这也就是网络的前向传播过程
    vgg.forward(x) 
    #将一个batch的数据喂入网络,得到网络的预测输出
    probablity=sess.run(vgg.prob,feed_dict={x:img_ready})
    #np.argsort(probablity[0])[-1:-6:-1]	
    top5=np.argsort[probablity][-1:-6:-1]
    print("top5:",top5)
    #定义两个list--对应的概率值和实际标签(zebra)
    values=[]
    bar_label=[]
    for n,i in enumerate(top5): #枚举上面取出的五个索引值
        print("n：",n)
        pritn("i:",i)
        #将索引值对应的预测概率值取出斌放入values
        values.append(probablity[0][i]) 
        bar_label.append(labels[i]) #根据索引值取出的实际标签并放入bar_label
        print(i,":",labels[i],"----",utils.percent(probablity[0][i]))
        #打印属于某个类别的概率
    ax=fig.add_subplot(111) #将画布划分为一行一列,并把下图放入其中
    ax.bar(range(len(values)),values,tick_label=bar_label,width=0.5,fc='g')
    ax.set_ylabel(u"probablity") #设置横轴标签
    ax.set_title(u"Top-5")
    for a,b in zip(range(len(values)),values):
        #在每个柱子的顶端添加对应的预测概率,a,b表示坐标,b+0.0005表示要把
        #文本信息放置在高于每个柱子顶端0.0005的位置
        #center是表示文本位于柱子顶端水平方向上的中间位置,bottom是将水平文本
        #放置在柱子顶端垂直方法上的底端位置,fontsize是字号
        ax.text(a,b+0.0005,utils.percent(b),ha="center",va="bottom",fontsize=7)
    plt.savefig("./result.jpg") #保存图片
    plt.show()#弹出展示图像