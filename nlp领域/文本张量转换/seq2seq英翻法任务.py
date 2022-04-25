# 导入open方法
from io import open
# 用于字符规范化
import unicodedata
# 正则表达式
import re
# 随即处理数据
import random
# 导入torch
import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入优化方法的工具包
from torch import optim
# 设备的选择
device = torch.device("cuda" if torch.cuda.is_available() else "cup")


# 数据预处理，将单词放和其对应的索引放到字典中
# 起始标志
SOS_token = 0
# 结束标志
EOS_token = 1

class Lang:
    def __init__(self, name):
        """初始化函数中参数name代表某种语言的名字"""
        # 将name传入类中
        self.name = name
        # 初始化词汇对应自然数值的字典
        self.word2index = {}
        # 初始化自然数值对应词汇字典，其中0和1对应的SOS和EOS已经在里面了。
        self.index2word = {0: "SOS", 1: "EOS"}
        # 初始化词汇对应的自然数索引，这里从2开始，因为0, 1已经被开始和结束标志占用了
        self.n_word = 2

    def addsentence(self, sentence):
        for word in sentence.split(" "):
            self.addword(word)

    """
    hello I am Jay
    word2index = {'hello': 2, 'I': 3, 'am': 4, 'Jay': 5}
    index2word = {0: 'SOS', 1: 'EOS', 2: 'hello', 3: 'I', 4: 'am', 5: 'Jay'}
    """
    def addword(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_word
            self.index2word[self.n_word] = word
            self.n_word += 1


# 字符规范化
# 将unicode转为Ascii,我们可以认为是去掉一些语言中的重音标记
def unicodetoascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Nn'
    )


def normalizestring(s):
    s = unicodetoascii(s.lower().strip())
    # 在.!?前加一个空格,\1表示配皮前面分组里面的任意一个，下面的意思就是将带有.!?里面其中一个的字符替换成" "+.!?
    s = re.sub(r"([.!?])", r" \1", s)
    # 使用正则表达式将字符串中不是大小写字母和正常表带你都替换成空格
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


data_path = '../data/eng-fra.txt'


def readlangs(lang1, lang2):
    lines = open(data_path, encoding='utf-8').read().strip().split('\n')
    pairs = [[normalizestring(s) for s in l.split('\t')]for l in lines]

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


# 设置组成句子中单词或标点的最多个数
MAX_LENGTH = 10

# 选择带有指定前缀的英文源语言的语句数据作为训练数据集
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


# 过滤语言对的具体逻辑函数
def filterpair(pair):
    return len(pair[0].split(' ')) < MAX_LENGTH and \
               pair[0].startswith(eng_prefixes) and \
                len(pair[1].split(' ')) < MAX_LENGTH


# 过滤语言对的具体逻辑函数
def filterpairs(pairs):
    return [pair for pair in pairs if filterpair(pair)]


def preparedata(lang1, lang2):
    input_lang, output_lang, pairs = readlangs(lang1, lang2)
    # 把符合条件的句子过滤出来（长度小于10， 以指定句子开头）
    pairs = filterpairs(pairs)
    for pair in pairs:
        input_lang.addsentence(pair[0])
        output_lang.addsentence(pair[1])
    return input_lang, output_lang, pairs


# 将语言对转换成模型输入需要的张量
def tensorfromsentence(lang, sentence):
    # 将句子按空格切分获得每个单词，再根据之前设置的字典获得每个单词对应的索引
    indexes = [lang.word2index[word] for word in sentence.split(' ')]
    # 加入句子结束标志
    indexes.append(EOS_token)
    # 将其使用torch.tensor封装成张量，并改变它的形状为n行1列，以方便后续计算
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorfrompair(pari):
    print(pari)
    input_tensor = tensorfromsentence(input_lang, pari[0])
    target_tensor = tensorfromsentence(output_lang, pari[1])
    return (input_tensor, target_tensor)


input_lang, output_lang, pairs = preparedata("eng", "fra")
pair_tensor = tensorfrompair(pairs[0])


# 构建GRU神经网络
class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        # 将参数hidden_size传入类中
        self.hidden_size = hidden_size
        # input_size是传入embedding第一个参数，也就是源语言词汇表大小
        # hidden_size:gru里面隐层维度要和词嵌入节点一致
        self.embedding = nn.Embedding(input_size, hidden_size, device=device)
        self.gru = nn.GRU(hidden_size, hidden_size, device=device)

    def forward(self, input1, hidden):
        # 将输入张量进行embedding操作，并转成三维
        output = self.embedding(input).view(1, 1, -1)
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def inithidden(self):
        # 初始化h0
        return torch.zeros(1, 1, self.hidden_size, device=device)

hidden_size = 25
input_size = 20
input = pair_tensor[0][0]
hidden = torch.zeros(1, 1, hidden_size, device=device)
encoder = EncoderRNN(input_size, hidden_size)
encoder_output, hidden = encoder(input, hidden)
print(encoder_output)

# 构建Decoder，加上注意力机制
class AttenDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttenDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self. max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)







