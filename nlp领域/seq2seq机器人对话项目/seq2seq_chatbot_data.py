# 导入库
import nltk
import itertools
import numpy as np
import pickle


# 定义白名单黑名单
EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz '
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''
# EN_BLACKLIST = 'ye'

FILENAME = './conv.txt'

limit = {
    'maxa': 18,
    'mina': 3,
    'maxq': 20,
    'minq': 0
}

UNK = 'unk'
GO = '<go>'
EOS = '<eos>'
PAD = '<pad>'
VOCAB_SIZE = 1000


def read_lines(filename):
    return open(filename, encoding='utf-8').read().split('\n')


def filter_line(line, EN_WHITELIST):
    j = []
    for ch in line:
        if ch in EN_WHITELIST:
            j.append(ch)
    return ''.join(j)

# def filter_line(line, whitelist):
#     return ''.join([ch for ch in line if ch in whitelist])
#     # return ''.join([ch for ch in line if ch in whitelist])

"""
这个是我自己写的方法，不能根据业务场景使用，假如question长度超过二十，answer长度没有超过二十，question和answer都要去掉
这个方法没有去掉answer。
def filter_data(line):
    qlines, alines = [], []
    for i in range(0, len(line)):
        if i % 2 == 0:
            if len([words for words in line[i].split(' ')]) >= limit['minq'] and len([words for words in line[i].split(' ')]) <= limit['maxq']:
                qlines.append(line[i])
        else:
            if len([words for words in line[i].split(' ')]) >= limit['mina'] and len([words for words in line[i].split(' ')]) <= limit['maxa']:
                alines.append(line[i])
    return qlines, alines
"""


def filter_data(line):
    qlines, alines = [], []
    for i in range(0, len(line), 2):
        qlen, alen = len(line[i].split(' ')), len(line[i + 1].split(' '))
        if qlen >= limit['minq'] and qlen <= limit['maxq']:
            if alen >= limit['mina'] and alen <= limit['maxa']:
                qlines.append(line[i])
                alines.append(line[i+1])
    return qlines, alines


def zero_pad(qtokenized, atokenized, w2idx):
    data_len = len(qtokenized)

    idx_q = np.zeros([data_len, limit['maxq']], dtype=np.int32)
    idx_a = np.zeros([data_len, limit['maxa'] + 2], dtype=np.int32)
    idx_o = np.zeros([data_len, limit['maxa'] + 2], dtype=np.int32)
    print(idx_a.shape)

    for i in range(data_len):
        q_indices = pad_seq(qtokenized[i], w2idx, limit['maxq'], 1)
        a_indices = pad_seq(atokenized[i], w2idx, limit['maxa'], 2)
        o_indices = pad_seq(atokenized[i], w2idx, limit['maxa'], 3)

        idx_q[i] = np.array(q_indices)
        idx_a[i] = np.array(a_indices)
        idx_o[i] = np.array(o_indices)

    return idx_q, idx_a, idx_o


def pad_seq(seq, w2idx, maxlen, flag):
    if flag == 1:
        indices = []
    elif flag == 2:
        indices = [w2idx[GO]]
    elif flag == 3:
        indices = []

    for word in seq:
        if word in w2idx:
            indices.append(w2idx[word])
        else:
            indices.append(w2idx[UNK])

    if flag == 1:
        return indices + [w2idx[PAD]] * (maxlen - len(indices))
    elif flag == 2:
        return indices + [w2idx[PAD]] * (maxlen - len(indices) + 2)

    elif flag == 3:
        return indices + [w2idx[EOS]] + [w2idx[PAD]] * (maxlen - len(indices) + 1)



def index_(tokenized_sentences, vocab_size):
    freq_dict = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # 根据次数从多到少[('词',次数)...*1000]
    vocab = freq_dict.most_common(vocab_size)

    print(vocab)
    index2word = [GO] + [EOS] + [UNK] + [PAD] + [x[0] for x in vocab]
    # ['<go>', '<eos>', 'unk', '<pad>', 'the', 'i', 'a', 'to', 'is', 'you', 'it'...]
    print(index2word)
    word2index = dict(zip(index2word, range(len(index2word))))
    print(word2index)
    # word2index = dict([(w, i) for i, w in enumerate(index2word)])
    return index2word, word2index, freq_dict


def process_data():
    # 读文件
    lines = read_lines(filename=FILENAME)
    # 将所有行转换成小写并存放在列表中
    lines = [line.lower() for line in lines]
    # 去掉黑名单的字符
    lines = [filter_line(line, EN_WHITELIST) for line in lines]
    # 分离出question，answer，返回列表[quesetion1,question2,...],[answer1,answer2,...]
    qlines, alines = filter_data(lines)

    # split()函数返回的本身就是个列表，外面再嵌套一个列表推导式，就是两层列表
    qtokenized = [wordlist.split(' ') for wordlist in qlines]

    # [['yeah', 'im', 'preparing', 'myself', 'to', 'drop', 'a', 'lot', 'on', 'this', 'man', 'but', 'definitely', 'need', 'something', 'reliable'], ['shouldnt', 'the', 'supporters', 'natural', 'answer', 'to', '

    atokenized = [wordlist.split(' ') for wordlist in alines]

    idx2w, w2idx, freq_dist = index_(qtokenized + atokenized, vocab_size=VOCAB_SIZE)

    # 保存列表和字典
    with open('idx2w.pkl', 'wb') as f:
        pickle.dump(idx2w, f)
    with open('w2idx.pkl', 'wb') as f:
        pickle.dump(w2idx, f)

    idx_q, idx_a, idx_o = zero_pad(qtokenized, atokenized, w2idx)
    print(idx_q.shape)

    # 保存了三个训练矩阵
    np.save('idx_q.npy', idx_q)
    np.save('idx_a.npy', idx_a)
    np.save('idx_o.npy', idx_o)

    # 保存里比较重要的信息
    metadata = {
        'w2idx': w2idx,
        'idx2w': idx2w,
        'limit': limit,
        'freq_dist': freq_dist
    }

    with open('metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

def load_data(PATH='.'):
    with open(PATH + 'metadata.pkl', 'rb') as f:
        metadata = pickle.load(f)

    idx_q = np.load(PATH + 'idx_q.npy')
    idx_a = np.load(PATH + 'idx_a.npy')

    return metadata, idx_q, idx_a



if __name__ == "__main__":
    process_data()