import tensorflow as tf
import numpy as np

# 创建0维张量
print(tf.constant(3))
# tf.Tensor(3, shape=(), dtype=int32)

# 创建一维张量
print(tf.constant([1, 2, 3]))
# tf.Tensor([1 2 3], shape=(3,), dtype=int32)

# 创建二维张量
print(tf.constant([[1, 2],[3, 4]]))
"""
tf.Tensor(
[[1 2]
 [3 4]], shape=(2, 2), dtype=int32)
"""
# 创建三维张量
print(tf.constant([
    [[1, 2],
     [3, 4]],
    [[4, 5],
     [6, 7]]
]))
"""
tf.Tensor(
[[[1 2]
  [3 4]]

 [[4 5]
  [6 7]]], shape=(2, 2, 2), dtype=int32)
"""

# 将张量转换成numpy
tensor1 = tf.constant([1, 2, 3, 4, 5])
print(np.array(tensor1))

print(tensor1.numpy())

# 张量运算常用函数
a = tf.constant([[1, 2],
                [3, 4]])
b = tf.constant([[1, 2],
                [1, 1]])
# 加法
print(tf.add(a, b))
"""
[[2 4]
 [4 5]], shape=(2, 2), dtype=int32)
"""

# 乘法
print(tf.multiply(a, b))
"""
[[1 4]
 [3 4]], shape=(2, 2), dtype=int32)
"""

# 矩阵乘法
print(tf.matmul(a, b))
"""
[[ 3  4]
 [ 7 10]], shape=(2, 2), dtype=int32)
"""

# 最大值
print(tf.reduce_max(a))
# tf.Tensor(4, shape=(), dtype=int32)
tf.reduce_mean
tf.reduce_sum
tf.reduce_min
# 最大值索引
print(tf.argmax(a))
# tf.Tensor([1 1], shape=(2,), dtype=int64)

# 变量，变量是一种特殊的参数，形状是不可改变的，但是其中的参数可以改变
var = tf.Variable([[1, 2], [3, 4]])
#print(var.shape())
# 改变参数前先变成numpy数组
var.numpy()
# 改变其中参数,但是不能改变他的形状
print(var.assign([[4, 5], [6, 7]]))


x1 = tf.Variable(3, name='x')
print(x1)