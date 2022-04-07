import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
# 构建神经网络模型
from tensorflow.keras.models import Sequential
# 相关网络层
from tensorflow.keras.layers import Dense,Dropout,Activation,BatchNormalization
# 导入辅助工具包
from tensorflow.keras import utils
# 正则化
from tensorflow.keras import regularizers
# 数据集
from tensorflow.keras.datasets import mnist


# 数据加载
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train)

# 创建画布，查看一下数据
plt.figure()
plt.imshow(x_train[1], cmap='gray')


