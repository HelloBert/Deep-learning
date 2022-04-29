import tensorflow as tf
import numpy as np


batch_size = 33
sequence_length = 20
hidden_size = 256
nums_layers = 2
num_encoder_symbols = 1004
num_decoder_symbols = 1004
embedding_size = 256
learning_rate = 0.001
model_dir = './model'

# 设置placeholder
encoder_input = tf.placeholder(tf.int32, shape=[batch_size, sequence_length])
decoder_input = tf.placeholder(tf.int32, shape=[batch_size, sequence_length])
target_output = tf.placeholder(tf.int32, shape=[batch_size, sequence_length])
weights_output = tf.placeholder(tf.float32, shape=[batch_size, sequence_length])


# 构造RNNcell网络单元
cell = tf.nn.rnn_cell.BasicLSTMCell(hidden_size)
cell = tf.nn.rnn_cell.MultiRNNCell([cell] * nums_layers)

# 导入之前数据预处理的数据
def loadQA():
    x_train = np.load('./idx_q.npy', mmap_mode='r')
    y_train = np.load('./idx_a.npy', mmap_mode='r')
    train_target = np.load('./idx_o.npy', mmap_mode='r')
    return x_train, y_train, train_target


# 用tensorflow给的框架构造网络
results, last_state = tf.contrib.legacy_seq2seq.embedding_rnn_seq2seq(
    tf.unstack(encoder_input, axis=1),
    tf.unstack(decoder_input, axis=1),
    cell,
    num_encoder_symbols,
    num_decoder_symbols,
    embedding_size,
    feed_previous = False
)

logtis = tf.stack(results, axis=1)

# 构造损失函数
loss = tf.contrib.seq2seq.sequence_loss(logtis, target_output, weights_output)

# 梯度下降优化算法
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# 初始化weights值
weights = np.ones(shape=[batch_size, sequence_length], dtype=np.float32)

# 起一个保存模型的Server
server = tf.train.Saver()

with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        server.restore(sess, ckpt.model_checkpoint_path)
    else:
        sess.run(tf.global_variables_initializer())

    eopch = 0
    for epoch in range(500000):
        eopch = epoch + 1
        print("epoch:", epoch)
        for step in range(0, 3):
            print("step:", step)
            train_x, train_y, train_target = loadQA()

            perm = np.arange(len(train_x))
            np.random.shuffle(perm)
            train_x = train_x[perm]
            train_y = train_y[perm]
            train_target = train_target[perm]

            encoder_input_train = train_x[step * batch_size: step * batch_size + batch_size, :]
            decoder_input_train = train_y[step * batch_size: step * batch_size + batch_size, :]
            target_output_train = train_target[step * batch_size: step * batch_size + batch_size, :]

            op, cast = sess.run([train_op, loss],
                        feed_dict={encoder_input: encoder_input_train,
                                decoder_input: decoder_input_train,
                                target_output: target_output_train,
                                weights_output: weights})
            step += 1
            print(cast)

        if epoch % 100 == 0:
            server.save(sess, model_dir + '/model.ckpt', global_step=epoch + 1)
























