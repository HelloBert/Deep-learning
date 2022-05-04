import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split

import unicodedata
import re
import numpy as np
import os
import io
import time

"""
# 下载文件
path_to_zip = tf.keras.utils.get_file("spa_eng.zip", origin="http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip", extract=True)
path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"


# 将unicode转成ascii
def unicode_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != "Mn")


def preprocessing_sentence(w):
    w = unicode_ascii(w.lower().strip())
    # ?.!,¿
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)
    w = re.sub(r"[^A-Za-z?.!,¿]+", " ", w)
    w = w.rstrip().strip()
    w = "<start>" + w + "<end>"
    return w


def create_dataset(path, num_examples):
    lines = io.open(path, encoding='utf-8').read().strip().split('\n')
    word_paris = [[preprocessing_sentence(w) for w in line.split('\t')] for line in lines[: num_examples]]
    return zip(*word_paris)


def tokenizer(lang):
    lang_Tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_Tokenizer.fit_on_texts(lang)
    tensor = lang_Tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_Tokenizer


def load_dataset(path, num_examples):
    targ_tensor, inp_tensor = create_dataset(path, num_examples)
    inp_tensor, inp_lang_Tokenizer = tokenizer(inp_tensor)
    targ_tensor, targ_lang_Tokenizer = tokenizer(targ_tensor)
    return inp_tensor, inp_lang_Tokenizer, targ_tensor, targ_lang_Tokenizer

num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)
"""
# 下载文件
path_to_zip = tf.keras.utils.get_file('spa_eng.zip', origin='http://storage.googleapis.com/download.tensorflow.org/data/spa-eng.zip', extract=True)
path_to_file = os.path.dirname(path_to_zip) + "/spa-eng/spa.txt"
print(os.path.dirname(path_to_zip))


# 将unicode文件转成ascii
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    w = re.sub(r'[^A-Za-z?.!,¿]+', " ", w)

    w.rstrip().strip()

    w = '<start>' + w + '<end>'
    return w

sp_sentence = u"¿Puedo tomar prestado este libro?"
# print(preprocess_sentence(sp_sentence))

def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    return zip(*word_pairs)

# 返回所有的英文放在一个元组中，所有西班牙语放在一个元组中
en, sp = create_dataset(path_to_file, None)


def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')

  """
 # 如果传入一个句子，将句子分成一个一个的字，例如"I love you"--》"I l o v e y o u",返回一个对象
 # 如果传入一个元组， 将元组里面的单词变成索引
"""
  # 实现分词
  lang_tokenizer.fit_on_texts(lang)

  # 将这个序列里面的字转换成向量，返回np数组
  tensor = lang_tokenizer.texts_to_sequences(lang)


  # 在后面进行填充0, 如果不填这个数字，那么会根据最长的序列的长度进行填充
  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')
  return tensor, lang_tokenizer

# lang = ("I love you", "hei nice to meet you")
# tensor, lang_tokenizer = tokenize(lang)

def load_dataset(path, num_examples=None):
    targ_lang, inp_lang = create_dataset(path, num_examples)
    # 传入一个元组，元组里面放的是一个一个句子， tokenize会将句子里面的单词转换成索引，每一句就是一个数组的一行，这个元组有多少句就有多少行

    # 传入英文
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    print(input_tensor.shape)
    # 传入西班牙文
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    print(target_tensor.shape)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer


def max_length(tensor):
    return max(len(t) for t in tensor)

# 限制数据集的大小以加快试验速度
"""
在超过 10 万个句子的完整数据集上训练需要很长时间。为了更快地训练，我们可以将数据集的大小限制为 3 万个句子（当然，翻译质量也会随着数据的减少而降低）
"""
num_examples = 30000
input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(path_to_file, num_examples)
max_length_targ, max_length_inp = max_length(target_tensor), max_length(input_tensor)
print(max_length_targ)
print(max_length_inp)


# 切分成训练集和测试集
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)


BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

print(dataset)
print(next(iter(dataset)))
#
# for inp, targ in dataset.take(steps_per_epoch):
#     print(inp.shape, targ.shape)

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))


class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # 隐藏层的形状 == （批大小，隐藏层大小）
    # hidden_with_time_axis 的形状 == （批大小，1，隐藏层大小）
    # 这样做是为了执行加法以计算分数
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # 分数的形状 == （批大小，最大长度，1）
    # 我们在最后一个轴上得到 1， 因为我们把分数应用于 self.V
    # 在应用 self.V 之前，张量的形状是（批大小，最大长度，单位）
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # 注意力权重 （attention_weights） 的形状 == （批大小，最大长度，1）
    attention_weights = tf.nn.softmax(score, axis=1)

    # 上下文向量 （context_vector） 求和之后的形状 == （批大小，隐藏层大小）
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # 用于注意力
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # 编码器输出 （enc_output） 的形状 == （批大小，最大长度，隐藏层大小）
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x 在通过嵌入层后的形状 == （批大小，1，嵌入维度）
    x = self.embedding(x)

    # x 在拼接 （concatenation） 后的形状 == （批大小，1，嵌入维度 + 隐藏层大小）
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # 将合并后的向量传送到 GRU
    output, state = self.gru(x)

    # 输出的形状 == （批大小 * 1，隐藏层大小）
    output = tf.reshape(output, (-1, output.shape[2]))

    # 输出的形状 == （批大小，vocab）
    x = self.fc(output)

    return x, state, attention_weights


encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


# def loss_function(real, pred):
#     mask = tf.math.logical_not(tf.math.equal(real, 0))
#     loss_ = loss_object(real, pred)
#
#     mask = tf.cast(mask, loss_.dtype)
#     loss_ *= mask
#     return tf.reduce_mean(loss_)

def loss_function(real, pred):
  # tf.math.equal(real, 0):将张量real中的所有0转成True,其他数字转成False
  # tf.math.logical_not(tf.math.equal(real, 0)):将张量中的Ture转成False, False转成True
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)


checkpoint_dir = "./training_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                 encoder=encoder,
                                 decoder=decoder)

@tf.function
def train_step(inp, targ, enc_hidden):
    print(inp.shape)
    print(targ.shape)
    print(targ)
    loss = 0
    with tf.GradientTape() as tape:
        enc_outputs, enc_hidden = encoder(inp, enc_hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        for t in range(0, targ.shape[1]):
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_outputs)
            loss += loss_function(targ[:, t], predictions)
            dec_input = tf.expand_dims(targ[:, t], 1)

    loss_batch = (loss / int(targ.shape[1]))
    # variables = encoder.trainabel_variables + decoder.trainable_variables
    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    return loss_batch

# @tf.function
# def train_step(inp, targ, enc_hidden):
#   loss = 0
#   with tf.GradientTape() as tape:
#     enc_output, enc_hidden = encoder(inp, enc_hidden)
#     dec_hidden = enc_hidden
#
#     dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)
#     # 教师强制 - 将目标词作为下一个输入
#     for t in range(1, targ.shape[1]):
#       # 将编码器输出 （enc_output） 传送至解码器
#       predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)
#       loss += loss_function(targ[:, t], predictions)
#
#       # 使用教师强制
#       dec_input = tf.expand_dims(targ[:, t], 1)
#
#   batch_loss = (loss / int(targ.shape[1]))
#
#   variables = encoder.trainable_variables + decoder.trainable_variables
#
#   gradients = tape.gradient(loss, variables)
#
#   optimizer.apply_gradients(zip(gradients, variables))
#
#   return batch_loss

"""
EPOCHS = 10

for epoch in range(EPOCHS):
    epoch_loss = 0
    enc_hidden = encoder.initialize_hidden_state()
    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        loss = train_step(inp, targ, enc_hidden)
        epoch_loss += loss

        if batch % 100 == 0:
            print("epoch:{}, batch:{}, loss:{:.4f}".format(epoch+1, batch, loss.numpy()))
    if (epoch + 1) % 2 == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)
    print("Epoch {} Loss {:.4f}".format(epoch+1, epoch_loss / steps_per_epoch))
"""

def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))

    sentence = preprocess_sentence(sentence)

    inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                           maxlen=max_length_inp,
                                                           padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden

    # 输入一行一列
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)
        #print(predictions[0])
        # 存储注意力权重以便后面制图
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()
        print(predicted_id)

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # 预测的 ID 被输送回模型
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot


# 注意力权重制图函数
def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 14}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict, rotation=90)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)

    print('Input: %s' % (sentence))
    print('Predicted translation: {}'.format(result))

    # attention_plot = attention_plot[:len(result.split(' ')), :len(sentence.split(' '))]
    # plot_attention(attention_plot, sentence.split(' '), result.split(' '))

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

translate(u'hace mucho frio aqui.')


