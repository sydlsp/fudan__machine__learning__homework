import warnings
import sklearn
import torch
from torch.utils import data
from torch import nn
import pandas as pd
import numpy as np
from sklearn import datasets
"""
写在前面：
波士顿房价数据集在调用过程中会报出警告：该数据集不建议使用
查阅资料得知其中可能涉及到有关于种族歧视的指标
本项目采用该数据集仅作学习使用
"""
warnings.filterwarnings("ignore")
"""
数据集的导入
"""

with warnings.catch_warnings():
    warnings.filterwarnings("ignore")
    train_features, train_labels =datasets.load_boston(return_X_y=True)

#波士顿房价数据集共有506条数据，有13个指标，预测的标签是房价
print(train_features.shape)
"""
对训练数据做归一化处理
"""
#这里先转化为pandas数组，好对特征进行归一化操作
train_features=pd.DataFrame(train_features)
#print(train_features.shape)

#归一化操作
train_features=train_features.apply(lambda x:(x-x.mean())/x.std())


"""
数据集的生成
"""

#先要把train_features转化成张量
train_features=torch.tensor(train_features.values,dtype=torch.float32)
train_labels=torch.tensor(train_labels,dtype=torch.float32)
dataset=data.TensorDataset(train_features,train_labels)
train_iter=data.DataLoader(dataset,batch_size=64,shuffle=True,drop_last=False)

"""
定义网络
"""

net=nn.Sequential(nn.Linear(train_features.shape[1],1))

"""
定义损失函数
"""
loss=nn.MSELoss()

optimizer=torch.optim.Adam(net.parameters(),lr=6,weight_decay=False)
"""
定义训练函数（就是纯训练）
"""
def train(net,train_iter,num_epoches,loss,optimizer):
    loss_list=[]
    for epoch in range(num_epoches):
        train_loss=0
        for X,y in train_iter:
            optimizer.zero_grad()
            y_hat=net(X)
            l=loss(y_hat,y)
            l.backward()
            optimizer.step()
            train_loss += l.item()
        loss_list.append(train_loss)
    return loss_list


"""
正式调用函数训练
"""
print(train(net,train_iter,15,loss,optimizer))


