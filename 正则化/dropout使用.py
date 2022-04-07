import tensorflow as tf
import numpy as np

# 定义dropout层
layer = tf.keras.layers.Dropout(0.2, input_shape=(2,))
# 定义输入数据
data = np.arange(1, 11).reshape(5, 2).astype(np.float32)
print(data)
# 对输入数据进行随机失活
outputs = layer(data, training=True)
print(outputs)