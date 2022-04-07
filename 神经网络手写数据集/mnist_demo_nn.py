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

"""
数据加载
数据处理
模型构建
模型训练
模型预测
模型保存
"""

# 数据加载
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train)
print(x_test.shape)

# 创建画布，查看一下数据
plt.figure()
plt.imshow(x_train[1], cmap='gray')
plt.show()


# 数据预处理
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255
x_test /= 255

# 将目标值转换成独热编码的形式
y_train = utils.to_categorical(y_train, 10)
y_test = utils.to_categorical(y_test, 10)

# 神经网络模型构建
model = Sequential()
# 全连接层:两个隐层，一个输出层
model = Sequential()
# 第一个隐层,512个神经元，先激活后BN,随机失活。
model.add(Dense(512, activation="relu", input_shape=(784,)))
model.add(BatchNormalization())
model.add(Dropout(0.2))
# 第二个隐层, 512个神经元，先BN，后激活，随机失活
# BN层和激活层没有先后顺序。
# Dropout一定是放在最后面的
model.add(Dense(512, kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Activation("relu"))
model.add(Dropout(0.2))
# 输出层
model.add(Dense(10, activation="softmax"))
print(model.summary())

# 模型编译
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(), metrics=tf.keras.metrics.Accuracy())

# 模型训练
history = model.fit(x_train, y_train, epochs=4, batch_size=128, validation_data=(x_test, y_test), verbose=1)
print(history.history)

# 模型评估
model.evaluate(x_test, y_test, verbose=1)

# 模型保存
model.save("model.h5")

# 模型加载
loadmodel = tf.keras.models.load_model("model.h5")
loadmodel.evaluate(x_test, y_test, verbose=1)

# 损失函数
plt.figure()
plt.plot(history.history['loss'], label="train_loss")
plt.plot(history.history['val_loss'], label="test_loss")
plt.legend()
plt.show()



