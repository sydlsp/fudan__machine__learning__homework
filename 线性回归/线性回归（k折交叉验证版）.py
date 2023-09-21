import warnings
import torch
from torch.utils import data
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

#这里train_features和train_labels是numpy类型的

"""
对训练数据归一化
"""
train_features=pd.DataFrame(train_features)#处理表格数据先转化成pandas类型
train_features=train_features.apply(lambda x:(x-x.mean())/(x.std()))

"""
在对表格数据处理完之后就已经能转化为张量了
"""
train_features=torch.tensor(train_features.values,dtype=torch.float32)
train_labels=torch.tensor(train_labels,dtype=torch.float32)

"""
下面做k折交叉验证工作：说白了就是把训练数据集给处理成 训练集和验证集
"""
def get_k_fold(k,i,train_features,train_labels):
    assert k>1
    fold_size=train_features.shape[0]//k
    X_train,y_train=None,None
    X_test,y_test=None,None
    for j in range(k):
        idx=slice(j*fold_size,(j+1)*fold_size)
        X_part,y_part=train_features[idx,:],train_labels[idx]
        if (j==i):
            X_test=X_part
            y_test=y_part
        elif X_train is None:
            X_train=X_part
            y_train=y_part
        else:
            X_train=torch.cat([X_train,X_part],0)
            y_train=torch.cat([y_train,y_part],0)

    return X_train,y_train,X_test,y_test

"""
到以上为止数据就可以说处理完了,定义训练函数,网络、损失函数和优化器写在训练函数里
"""
def train_and_test (train_features,train_labels,test_features,test_labels,num_epoches,learning_rate,batch_size):
    dataset=data.TensorDataset(train_features,train_labels)
    train_iter=data.DataLoader(dataset,batch_size=batch_size,shuffle=True)
    net=nn.Sequential(nn.Linear(train_features.shape[1],1))
    loss=nn.MSELoss()
    optimizer=torch.optim.Adam(net.parameters(),lr=learning_rate,weight_decay=False)
    train_loss_list=[]
    test_loss_list=[]#用于记录训练误差，有评价指标的意思
    for epoch in range(num_epoches):
        for X,y in train_iter:
            optimizer.zero_grad()
            l=loss(net(X),y)
            l.backward()
            optimizer.step()
        train_loss_list.append(l.item())
        test_loss_list.append(loss(net(test_features),test_labels).item())
    #训练可视化部分
    list_x=[i for i in range(0,num_epoches,1)]
    plt.figure()
    plt.xlabel("epoch")#设置坐标轴名称
    plt.ylabel("MseLoss")
    plt.xlim((0,num_epoches-1))#设置坐标轴范围
    plt.xticks(range(0,num_epoches,1))  #设置坐标轴刻度也就是一格前进多少
    # 若属性用的是全名则不能用 * fmt * 参数来组合赋值，应该用关键字参数对单个属性赋值如：
    # plot(x, y2, color='green', marker='o', linestyle='dashed', linewidth=1, markersize=6)
    # plot(x, y3, color='#900302', marker='+', linestyle='-')
    plt.plot(list_x, train_loss_list,color='blue',label='train_loss')
    plt.plot(list_x, test_loss_list,color='orange',label='test_loss')
    #加图例
    plt.legend()
    plt.show()

    return train_loss_list,test_loss_list

"""
整体函数
"""
def train_k_fold(k,train_features,train_labels,num_epoches,learning_rate,batch_size):
    for i in range(k):
        X_train,y_train,X_test,y_test=get_k_fold(k,i,train_features,train_labels)
        print("第i折交叉验证"+str(i))
        print(train_and_test(X_train,y_train,X_test,y_test,num_epoches,learning_rate,
                       batch_size))

"""
调用整体函数，真正训练
"""

train_k_fold(5,train_features,train_labels,10,1,64)










