import torch
import torchvision
from torch import nn
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
定义网络
"""
class Net(nn.Module):
    def __init__(self,num_inputs,num_outputs,num_hiddens1,is_training=True):
        super().__init__()
        self.num_inputs=num_inputs
        self.out_puts=num_outputs
        self.num_hiddens1=num_hiddens1
        self.training=is_training
        self.lay1=nn.Linear(self.num_inputs,self.num_hiddens1)
        self.lay2=nn.Linear(self.num_hiddens1,self.out_puts)
        self.dropout=nn.Dropout(p=0.5)
        self.relu=nn.ReLU()

    def forward(self,x):
        H1=self.relu(self.lay1(x.reshape(-1,self.num_inputs)))
        if self.training==True:
            H1=self.dropout(H1)
        H2=self.lay2(H1)
        return H2

"""
定义预测函数并计算准确度
"""
def predict(net,test_data_loader,device):
    net.eval()
    net.to(device)
    accuracy=[]
    for x,y in test_data_loader:
        x,y=x.to(device),y.to(device)
        x_predict,x_index=torch.max(net(x),dim=1)
        equal_list=(x_index==y).int()
        batch_accuracy=sum(equal_list)/len(equal_list)
        accuracy.append(batch_accuracy)
    return (sum(accuracy)/len(accuracy)).item()

"""
定义训练过程，每一轮训练完查看在测试集上准确率
"""
def train(net,train_iter,test_iter,batch_size,num_epoches,device,lr=0.1):
    net.train()
    net.to(device)
    loss=nn.CrossEntropyLoss()
    optimizer=torch.optim.SGD(net.parameters(),lr=lr)
    for epoch in range(num_epoches):
        print("epoch:",epoch)
        train_loss=[]
        for x,y in train_iter:
            x,y=x.to(device),y.to(device)
            optimizer.zero_grad()
            y_hat=net(x)
            l=loss(y_hat,y.long())  #CrossEntropyLoss要求是long型数
            #print(l)
            l.backward()
            optimizer.step()
            train_loss.append(l.item())
        #print(train_loss)
        print("训练损失",round(sum(train_loss)/len(train_loss),3))
        print("训练集准确率",round(predict(net,train_iter,device),3))
        print("测试集准确率",round(predict(net,test_iter,device),3))  #设置一下输出的小数位数



net=Net(784,10,256)
num_epoches,batch_size=10,128
device=torch.device("cuda:0"  if torch.cuda.is_available() else "cpu")
train(net,train_data_loader,test_data_loader,batch_size=batch_size,num_epoches=num_epoches,device=device)


"""
实践证明不同的损失函数对训练效果影响很大，分类问题还是交叉熵损失函数好
"""









