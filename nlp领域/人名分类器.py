"""
人名分类器任务
文件名就是国家的名字
文件中的内容就是本国的人名

导入包
下载数据，对数据清洗
构建RNN模型（LSTMGRU）
构建训练函数并进行训练
构建评估函数
"""


from io import open
# 帮助使用正则表达式进行子目录查询
import glob
import os
# 用于常见字母及字符规范化
import string
import unicodedata
# 随机工具
import random
# 导入时间和数学工具包
import time
import math
# 导入torch工具
import torch
# 导入nn准备构建模型
import torch.nn as nn
# 引入制图工具包
import matplotlib.pyplot as plt

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)
print(n_letters)


# 去掉一些语言中的重音标记
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn'
                   and c in all_letters)


data_path = './data/names/'

# 读取每个文件，将每个文件转换成行，放入unicodeToAscii函数中去除重音符号。
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

filename = data_path + 'Chinese.txt'
result = readLines(filename)
print(result[:20])

# 构建一个人名类别列表与人名对应关系的字典
category_lines = {}

all_category = []

n_category = 0
# 读取指定路径下的txt文件，使用glob,path中可以使用正则表达式
for filename in glob.glob(data_path + '*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_category.append(category)
    n_category += 1
    # 读取每个文件，形成名字列表
    Lines = readLines(filename)
    category_lines[category] = Lines
print(n_category)
print(all_category[4])


def lineToTensor(line):
    # 初始化一个全0的张量，这个张量的形状是(len(line),1,n_letters)
    # 代表人名中的每个字母都用一个(1*n_letters)张量表示
    tensor = torch.zeros(len(line), 1, n_letters)
    # print(tensor)
    # 遍历每个人名中的每个字符，并搜索其对应的索引，将其索引位置置1
    for li, letter in enumerate(line):
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


# 定义网络
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_zise, num_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_zise
        self.num_layers = num_layers
        # 实例化RNN层
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers)
        # 定义全连接层,作用是将RNN输出维度转换成指定输出维度
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        # 实例化softmax层,从输出层中获取类别的结果
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input1, hidden):
        # input1代表任命分类器中的输入张量，形状是1*letters
        # hidden代表rnn隐藏层张量，形状self.num_layers * 1 * self.hidden_size
        # 输入到RNN中的张量要求是三维，所以要用unsqueeze()函数扩充张量
        input1 = input1.unsqueeze(0)
        # 将数据和隐层张量扔到网络中
        rr, hn = self.rnn(input1, hidden)
        # 将从rnn中获得的结果传输进全连接层
        return self.softmax(self.linear(rr)), hn

    def initHidden(self):
        # 初始化一个全零的隐藏层张量，维度是3
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# 定义LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        # 实例化LSTM层
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        # 实例化全连接层
        self.linera = nn.Linear(self.hidden_size, self.output_size)
        # 实例化softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    # LSTM输入有三个张量
    def forward(self, input1, hidden, c):
        input1 = input1.unsqueeze(0)
        output, (hn, cn) = self.lstm(input1, (hidden, c))
        return self.softmax(self.linera(output)), hn, cn

    def initHiddenAndC(self):
        hidden = c = torch.zeros(self.num_layers, 1, self.hidden_size)
        return hidden, c


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        # 实例化全连接层
        self.liearn = nn.Linear(self.hidden_size, self.output_size)
        # 实例化softmax
        self.softmax = nn.LogSoftmax(dim=-1)
        # 实例化GRU层
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers)

    def forward(self, input1, hidden):
        input1 = input1.unsqueeze(0)
        output, hn = self.gru(input1, hidden)
        output = self.softmax(self.liearn(output))
        return output, hn

    def initHidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


# 参数
input_size = n_letters
n_hidden = 128
output_size = n_category

input1 = lineToTensor('B').squeeze(0)

hidden = c = torch.zeros(1, 1, n_hidden)

rnn = RNN(input_size, n_hidden, output_size)
lstm = LSTM(input_size, n_hidden, output_size)
gru = GRU(input_size, n_hidden, output_size)

rnn_output, next_hidden = rnn(input1, hidden)
print("rnn:", rnn_output)
print("rnn_shape", rnn_output.shape)


def categoryFromOutput(output):
    # 从输出结果中的到指定的类别
    # 需要调用topk函数，得到最大值和其对应的索引，作为我们的类别信息
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    # 从前面已经构建好的all_category获取索引所对的值
    return all_category[category_i], category_i


category, category_i = categoryFromOutput(rnn_output)
print(category, category_i)


# 随机生成训练数据
def randomTrainExample():
    # 随机获取一个国家
    category = random.choice(all_category)
    # 从这个国家中随机获取一个名字
    line = random.choice(category_lines[category])
    # 获取名字对应的索引值，转成tensor
    category_tensor = torch.tensor([all_category.index(category)], dtype=torch.long)
    # 将tensor转成One-hot编码
    line_tensor = lineToTensor(line)
    # 返回随机一个国家， 国家对应一个人名， 国家索引张量，人名One-hot编码
    return category, line, category_tensor, line_tensor


# 随机获取十个国家， 国家对应一个人名， 国家索引张量，人名One-hot编码
for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainExample()
    print(category, line, category_tensor, line_tensor.size())


# 构建传统的RNN训练函数
# 定义损失函数
criterion = nn.NLLLoss()

# 设置学习率为0.005
learning_rate = 0.005

def trainRNN(category_tensor, line_tensor):
    # category_tensor是训练数据的标签
    # line_tensor是训练数据的特征
    # 初始化h0
    hidden = rnn.initHidden()

    # 关键的一步，将模型结构中的梯度归零
    rnn.zero_grad()

    # 循环遍历训练数据中line_tensor中的每一个字符，传入RNN中，并且迭代更新hidden
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)

    # 因为RNN的输出是三维张量,为了满足category_tensor, 需要进行降维操作
    loss = criterion(output.squeeze(0), category_tensor)

    # 进行反向传播
    loss.backward()

    # 现实更新模型中所有的参数
    for p in rnn.parameters():
        # 将参数的张量标识与参数的梯度进行乘法运算并乘以学习率，结果加到参数上，并进行覆盖更新。
        p.data.add_(-learning_rate * p.grad.data)

    # 返回RNN最终的输出结果output和模型的损失loss
    return output, loss.item()


# 构建LSTM训练函数
def trainLSTM(category_tensor, line_tensor):
    hidden, c = lstm.initHiddenAndC()
    lstm.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hn, cn = lstm(line_tensor[i], hidden, c)
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()
    for p in lstm.parameters():
        p.data.add_(-learning_rate * p.grad.data)
    return output, loss.item()


# 构建GRU训练函数
def trainGRU(category_tensor, line_tensor):
    hidden = gru.initHidden()
    gru.zero_grad()
    for i in range(line_tensor.size()[0]):
        output, hidden = gru(line_tensor[i], hidden)
    loss = criterion(output.squeeze(0), category_tensor)
    loss.backward()
    for p in lstm.parameters():
        p.data.add_(-learning_rate * p.grad.data)
    return output, loss.item()


# 构建时间计算函数
def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m*60
    # 返回指定格式的耗时
    return '%dm %ds' % (m, s)
since = time.time() - 10 * 60
period = timeSince(since)
print(period)

# 设置迭代次数
n_iters = 10000
# 设置结果打印间隔
print_every = 50
# 设置绘图损失函数上的制图间隔
plot_every = 10


# 定义训练函数
def train(train_type_fn):
    # 每个制图间隔损失保存列表
    all_losses = []
    # 获得训练开始时间戳
    start = time.time()
    # 设置初始间隔损失函数为0
    current_loss = 0
    # 从1开始进行训练迭代，共n_iters次
    for iter in range(1, n_iters+1):
        # 通过randomTrainingExample函数随机获取一组训练数据和标签
        category, line, category_tensor, line_tensor = randomTrainExample()
        # 将训练数据和对应类别标签张量表示传入到train函数中
        output, loss = train_type_fn(category_tensor, line_tensor)
        # 计算制图间隔中的总损失
        current_loss += loss
        # 如果迭代数能够整除打印间隔
        if iter % print_every == 0:
            # 获取该迭代步上的output通过categoryFromOutput函数获得对应的类别和类别索引
            guess, guess_i = categoryFromOutput(output)
            # 和真实的类别category作比较，如果相同则打对号，否则打❌
            correct = '√' if guess == category else '❌(%s)' % category
            # 打印迭代步数，迭代步百分比，当前训练耗时，损失，该步预测的名字，以及是否正确
            print('%d %d%% (%s) %.4f / %s %s' %(iter, iter/n_iters * 100, timeSince(start), loss, guess, correct))

        # 如果迭代能够整除制图间隔
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            # 间隔重置为0
            current_loss = 0
    # 返回对应总损失列表和训练耗时
    return all_losses, int(time.time() - start)

# 调用train函数，分别进行RNN，LSTM，GRU模型训练
all_losses1, period1 = train(trainRNN)
all_losses2, period2 = train(trainLSTM)
all_losses3, period3 = train(trainGRU)


# 绘制损失对比曲线，训练耗时对比柱状图
plt.figure(0)
plt.plot(all_losses1, label='RNN')
plt.plot(all_losses2, color='red', label='LSTM')
plt.plot(all_losses3, color='orange', label='GRU')
plt.legend(loc='upper left')

# 创建耗时柱状图画布
plt.figure(1)
x_data = ['RNN', 'LSTM', 'GRU']
y_data = [period1, period2, period3]
plt.bar(range(len(x_data)), y_data, tick_label=x_data)
plt.show()



# 构建评估函数
def evaluateRNN(line_tensor):
    # 初始化一个隐藏层张量
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = rnn(line_tensor[i], hidden)
    return output.squeeze(0)


def evaluateLSTM(line_tensor):
    # 初始化一个隐藏层张量
    hidden, c = lstm.initHiddenAndC()
    for i in range(line_tensor.size()[0]):
        output, hn, cn = lstm(line_tensor[i], hidden, c)
    return output.squeeze(0)


def evaluateGUR(line_tensor):
    # 初始化一个隐藏层张量
    hidden = gru.initHidden()
    for i in range(line_tensor.size()[0]):
        output, hidden = gru(line_tensor[i], hidden)
    return output.squeeze(0)

# 构建预测函数
def predict(input_line, evaluate_fn, n_predictions=3):
    """
    :param input_line: 代表输入字符串的名字
    :param evaluate_fn: 代表评估模型函数，RNN，LSTM，GRU
    :param n_predictions: 代表最有可能的n_predictions个结果
    :return:
    """
    # 首先将输入的名字打印出来
    print('\n> %s' % input_line)
    # 注意:以下操作相关张量不进行求梯度
    with torch.no_grad():
        # 是输入的名字张量表示，并使用evaluate函数获得预测输出
        output = evaluate_fn(lineToTensor(input_line))

        # 从预测结果中取top3个最大值及其索引
        topv, topi = output.topk(n_predictions, 1, True)

        # 初始化列表
        predictions = []
        for i in range(n_predictions):
            # 从topv中取出概率值
            value = topv[0][i].item()
            # 从topi中取出索引值
            category_index = topi[0][i].item()
            # 打印概率值以及对应的真实国家名称
            print('(%.2f) %s' % (value, all_category[category_index]))
            predictions.append([value, all_category[category_index]])
        return predictions


for evaluate_fn in [evaluateRNN, evaluateLSTM, evaluateGUR]:
    print('----------------------')
    predict('Dovesky', evaluate_fn)
    predict('Jackson', evaluate_fn)
    predict('Satoshi', evaluate_fn)








