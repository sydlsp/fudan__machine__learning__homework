import numpy as np
import torch

A=torch.arange(0,9).reshape(3,3)
print (A)
B=torch.tensor([1,2,3],dtype=torch.float32)
print (B)
print (A-B)
C=torch.tensor([[1,2,3],[4,5,6]],dtype=torch.float32)

test_A=torch.tensor([[1,2,3],[4,5,6]]).numpy()
test_B=torch.tensor([1,2,3]).numpy()
test_C=test_A-test_B
print(np.sum(test_A-test_B,axis=1))



