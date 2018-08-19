import numpy as np  
import h5py  
def load_dataset():  
	#训练集
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")  #读取训练数据，共209张图片
    #提取特征值
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  #原始训练集（209*64*64*3）  
    #提取标签
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  #原始训练集的标签集（y=0非猫,y=1是猫）（209*1）  
    #测试集
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")  #读取测试数据，共50张图片
    #提取特征值
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  #原始测试集（50*64*64*3）  
    #提取标签
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  #原始测试集的标签集（y=0非猫,y=1是猫）（50*1）
    #[b'non-cat' b'cat']	
    classes = np.array(test_dataset["list_classes"][:])  # the list of classes  
	#shape[0]取矩阵的行数
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))  #原始训练集的标签集设为（1*209）
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))  #原始测试集的标签集设为（1*50）
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes  

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes=load_dataset()   
print(classes)   
print(len(classes))