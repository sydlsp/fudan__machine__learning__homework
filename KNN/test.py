import numpy
import numpy as np

array=[0,0,1,4,5,6,7,8,9,9,9]
print(np.argmax(np.bincount(array)))

list_A=[1,2,3]
list_A=np.array(list_A)
list_B=[1,5,9]
list_B=np.array(list_B)
print((list_A==list_B).astype(np.int))

list_C=[1,4,6,7,8,0]
list_C=np.array(list_C)
print(list_C)
print(list_C.argsort())
print("ok")