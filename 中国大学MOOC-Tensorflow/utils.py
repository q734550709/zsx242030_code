#coding:utf-8
from skimage import io,transform
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from pylab import mpl

mpl.rcParams['font.sans-serif']=['SimHei'] #正常显示中文标签
mpl.rcParams['axes.unicode_minus']=False #正常显示正负号

def load_image(path):
    fig=plt.figure("Center and Resize")
    img=io.imread(path)#根据传入的路径读图片
    img=img/255.0 #将像素归一化到[0,1]
    #将该画布分为一行三列
    ax0=fig.add_subplot(131)
    ax0.set_xlabel(u"Original Picture") #添加子标签
    ax0.imshow(img) #添加展示该图像
    short_edge=min(img.shape[:2])  #找到该图像的最短边
    y=int((img.shape[0]-short_edge)/2)
    x=int((img.shape[1]-short_edge)/2) #把图像的w和h分别减去最短边,并求平均值
    crop_img=img[y:y+short_edge,x:x+short_edge] #取出切分出的中心图像
    print(crop_img.shape)
    
    ax1=fig.add_subplot(132) #把下面的图像放在该画布的第二个位置
    ax1.set_xlabel(u'Center Picture') #添加子标签
    ax1.imshow(crop_img) 
    
    re_img=transform.resize(crop_img,(224,224)) #resize成固定的image_size
    ax2=fig.add_subplot(133) #把下面的图像放在该画布的第三个位置	
    ax2.set_xlabel(u"Resize Picture") #添加子标签
    ax2.imshow(re_img)
    img_ready=re_img.reshape((1,224,224,3))
    return img_ready
    
def percent(value):
    return "%.2%%"%(value*100)