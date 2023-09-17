import torch

A=torch.tensor([1,9,5,3,8,3,0],dtype=torch.float32)
a,idx=torch.sort(A,descending=True)  #descending表示降序  a是降序排列的结果 idx是在原来数组中的下标 idx[0]=1表示最大的元素是原来数组的A[1]
print(a)
print(idx)