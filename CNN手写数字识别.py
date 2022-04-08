import tensorflow as tf
from tensorflow.keras.datasets import mnist

# 导入数据
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 数据处理，做维度调整
train_images = tf.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
test_images = tf.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1 ))

# 模型构建
net = tf.keras.models.Sequential([
    # 卷积层 6个5*5的卷积核 sigmoid激活函数
    tf.keras.layers.Conv2D(filters=6, kernel_size=5, activation="sigmoid", input_shape=(28, 28, 1)),
    # 池化层（最大池化）
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    # 卷积层 16 5*5 sigmoid
    tf.keras.layers.Conv2D(filters=16, kernel_size=5, activation="sigmoid"),
    # 池化层 （最大池化）
    tf.keras.layers.MaxPool2D(pool_size=2, strides=2),
    # 接全连接层之间将数据抻平
    tf.keras.layers.Flatten(),
    # 全连接层
    tf.keras.layers.Dense(120, activation="sigmoid"),
    # 全连接层
    tf.keras.layers.Dense(84, activation="sigmoid"),
    # 输出层
    tf.keras.layers.Dense(10, activation="softmax")
])
print(net.summary())

# 模型编译
net.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.9),
            loss=tf.keras.losses.sparse_categorical_crossentropy,
            metrics=["accuracy"])

# 训练模型
net.fit(train_images, y_train, epochs=5, batch_size=128, verbose=1)

# 模型评估
print("模型评估分数")
net.evaluate(test_images, y_test, verbose=1)


