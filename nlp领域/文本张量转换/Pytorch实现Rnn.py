import torch
import torch.nn as nn

# W矩阵5*6，第一个数字是第一层输出神经元个数，第二个数字是隐藏层神经元个数，第三个数字隐藏层数量
rnn = nn.RNN(input_size=100, hidden_size=20, num_layers=2)
# 第一个参数:每个句子单词数量，第二个参数:输入矩阵3批次(3个句子)，第三个参数:5个神经元，3*5
X = torch.randn(5, 3, 100)

# 送入当前输入的X,h0是初始化的矩阵，不给的话是0
# h0,[隐藏层数量, batch, 特征长度]
output, ht = rnn(X, torch.zeros(2, 3, 20))
"""
return:
ht是最后时刻加和的返回
out是指每个时刻的输出
"""
print(output)
print(output.shape)
print(ht)
print(ht.shape)