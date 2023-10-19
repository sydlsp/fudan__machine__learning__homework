#导入所需的模块
import time
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from keras.datasets import mnist
from sklearn import svm


"""
数据集的读取和处理
"""
#导入Keras提供的数据集MNIST模块
(x_train,y_train), (x_test,y_test) = mnist.load_data()

#转化（reshape）为一维向量，其长度为784，并设为Float数。
x_Train =x_train.reshape(60000, 784).astype('float32')
x_Test = x_test.reshape(10000, 784).astype('float32')

#将数据归一化
x_Train_normalize = x_Train / 255
x_Test_normalize = x_Test / 255

"""
定义SVM模型
"""
#传递训练模型的参数
print(time.strftime('%Y-%m-%d %H:%M:%S'))
net= svm.SVC(C=100.0, kernel='rbf', gamma=0.03)  #惩罚系数为100，采用高斯核函数，核函数系数为0.03


"""
模型训练
"""
# 进行模型训练
t1 = time.time()
net.fit(x_Train_normalize, y_train)
t2 = time.time()
SVMfitime = float(t2-t1)
print("Time taken: {} seconds".format(SVMfitime))


"""
进行预测
"""
predictions = [int(a) for a in net.predict(x_Test_normalize)]

"""
指标计算
"""
#f1-score,precision,recall
print(classification_report(y_test, np.array(predictions)))

#计算准确率
print('accuracy=', accuracy_score(y_test, predictions))
print(time.strftime('%Y-%m-%d %H:%M:%S'))


