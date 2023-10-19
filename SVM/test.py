import torch

a=torch.tensor([1,2,3,4],dtype=torch.float32)
b=torch.tensor([1,3,4,5],dtype=torch.float32)
print((a==b).int())