import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

opt = tf.keras.optimizers.SGD(learning_rate=0.1)
var = tf.Variable(1.0)
# 定义损失函数
loss = lambda:(var**2)/2.0
# 计算梯度，并进行参数更新
print(opt.minimize(loss, [var]))
# 参数更新结果
print(var.numpy())

# 这里使用梯度下降法公式Wt = W(t-1) - 学习率*梯度
# 学习率已知，梯度是对自己定义的loss损失函数求导得到var
# 那么结果就是1.0 - 0.1*1.0 = 0.9

# 实现动量的梯度下降算法
opt = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
# 定义要更新的参数
var = tf.Variable(1.0)
var0 = var.value()
# 定义损失函数
loss = lambda :(var**2)/2.0
# 第一次更新
opt.minimize(loss, [var])
var1 = var.value()
# 第二次更新
opt.minimize(loss, [var])
var2 = var.value()
print(var0, var1, var2)
# 1.0, 0.9, 0.71
print("步长:", var0-var1)
print("步长:", var1-var2)
# 步长: tf.Tensor(0.100000024, shape=(), dtype=float32)
# 步长: tf.Tensor(0.18, shape=(), dtype=float32)

# 实现adagrad梯度下降方法
# 实例化
opt = tf.keras.optimizers.Adagrad(learning_rate=0.1, initial_accumulator_value=0.1, epsilon=1e-06)
var = tf.Variable(1.0)
def loss():
    return (var**2)/2.0
# 更新
opt.minimize(loss, [var])
print(var)

# 实现RMS梯度下降方法
# 实例化
opt = tf.keras.optimizers.RMSprop(learning_rate=0.1, rho=0.9)
var = tf.Variable(1.0)
def loss():
    return (var**2)/2.0
# 更新
opt.minimize(loss, [var])
print(var)

# Adam梯度下降方法
opt = tf.keras.optimizers.Adam(learning_rate=0.1)
var = tf.Variable(1.0)
def loss():
    return (var**2)/2.0
# 更新
opt.minimize(loss, [var])
print(var)

