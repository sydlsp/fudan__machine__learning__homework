import torch
import torchvision
import numpy as np
from torchvision import datasets,transforms
from torch.utils.data import DataLoader,Dataset

"""
MINST数据集的下载
"""

train_data=datasets.MNIST(root="./data/",train=True,transform=transforms.ToTensor(),download=True)

test_data=datasets.MNIST(root="./data/",train=False,transform=transforms.ToTensor(),download=True)

"""
MINST数据集的加载
"""

train_data_loader=torch.utils.data.DataLoader(dataset=train_data,batch_size=64,shuffle=True,drop_last=True)
test_data_loader=torch.utils.data.DataLoader(dataset=test_data,batch_size=64,shuffle=False,drop_last=False)

"""
对训练数据和测试数据进行处理
"""
def getmean(x_train):  #其实是得到了60000张图的一张平均图，为下面的归一化操作做好准备
    x_train=np.reshape(x_train,(x_train.shape[0],-1)) #把28*28像素展开为一行，因为minst数据集测试集有60000张图，所以在本例中有60000行
    mean_image=np.mean(x_train,axis=0)
    return mean_image

def centralized (x_train,mean_image):
    x_train=x_train.reshape(x_train.shape[0],-1)
    x_train=x_train.astype(float)
    x_train-=mean_image
    return x_train



#对训练数据进行处理
x_train=train_data_loader.dataset.data.numpy()  #这里返回的是(60000,28,28)
y_train=train_data_loader.dataset.targets.numpy() #这里返回的是(60000,)
#归一化处理
mean_image=getmean(x_train)
x_train=centralized(x_train,mean_image)


#对测试数据进行处理

num_test=200
x_test=test_data_loader.dataset.data[:num_test].numpy()  #这里返回的是(60000,28,28)张量
y_test=test_data_loader.dataset.targets[:num_test].numpy() #这里返回的是(60000,)张量

#归一化处理
mean_image=getmean(x_test)
x_test=centralized(x_test,mean_image)


def knn(x_train,x_test,y_train,k):
    num_test=x_test.shape[0]
    label_list=[]
    for i in range(num_test): #对测试集中的每一个点分别进行计算
        predict_label=[]
        distances=np.sqrt(np.sum((x_train-x_test[i])**2,axis=1))#distance矩阵是待归类点到训练点的距离
        idxs=distances.argsort()[:k]#对distance数组进行排序，返回排序索引，argsort函数是按从小到大排列

        # predict_label=np.argmax(np.bincount(idxs))
        for j in idxs:
            predict_label.append(y_train[j]) #把索引对其到label上,predict_label实际上存放了前k个点所属的类别

        label_list.append(np.argmax(np.bincount(predict_label))) #找predict_label中出现最多的加入label_list
    return label_list

def predict_accuracy(label_list,y_test):  #预测准确度计算
    count_list=(label_list==y_test).astype(float)
    return float((count_list.sum()/len(count_list)))*100

key_value_list=[1,3,5]
for k_value in key_value_list:
    print("k="+str(k_value)+"   "+str(predict_accuracy(knn(x_train,x_test,y_train,k_value),y_test))+"%")



