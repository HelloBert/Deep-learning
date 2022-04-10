"""
CIFAR数据集，图像大小：3*32*32
十类别
步骤:
 1.下载CIFAR数据集
 2.构建神经网络分类
 3.定义损失函数
 4.训练神经网络数据集，并进行验证
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
# 这个数据集是[0, 1]之间，我们要将其归一化到[-1, 1]之间，transforms.Normalize标准化，前面传入均值，后面参数是方差，
# 这样就将数据标准化成为[-1, 1]之间的数据了
# num_workers=2 两个进程进行处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=0)

classes = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")

# 构建展示图片的函数
def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (2, 1, 0)))
    plt.show()

"""
# 从数据迭代器中读取一张图片
dataiter = iter(trainloader)
images, label = dataiter.next()
print("=====images======")
print(images)
print("======label======")
print(label)
# 展示图片,torchvision.utils.make_grid:将四张图片合并
imshow(torchvision.utils.make_grid(images))

# 打印标签
print(' '.join('%5s' % classes[label[i]] for i in range(4)))
"""

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义卷积层
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 定义池化层
        self.pool = nn.MaxPool2d(2, 2)
        # 定义全连接层
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 变换x的形状以适配全连接输入
        x = x.view(-1, self.num_flat_feature(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_feature(self, x):
        size = x.size()[1:]
        num_feature = 1
        for i in size:
            num_feature *= i
        return num_feature


# net = Net()
# print(net)
"""

# 定义梯度下降算法优化器
opt = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# 编写训练代码
for epoch in range(2):
    running_loss = 0.0
    # 按批次迭代训练数据
    for i ,data in enumerate(trainloader, 0):
        # 从data中去除含有输入图像的张量input, 标签labels
        inputs, labels = data

        # 第一步将梯度清零
        opt.zero_grad()

        # 将输入数据输入到神经网络中， 得到输出张量
        outputs = net(inputs)

        # 计算损失值
        loss = criterion(outputs, labels)

        # 进行反向传播
        loss.backward()

        # 更新参数
        opt.step()

        # 打印训练信息
        running_loss += loss.item()
        if (i + 1) % 2000 == 0:
            # 每轮每两千个数据打印一下损失值
            print("[%d, %5d] loss: %3f" %(epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0

print("Finfished Training.")

# 设置模型保存位置
PATH = './crfar_net.pth'
# 保存模型状态字典，也就是模型里面的参数
torch.save(net.state_dict(), PATH)
"""


# 测试

"""
# 展示几张图片作为预测数据对比
dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# 加载模型只对一批次数据（4张图片）进行预测
# 实例化
net = Net()
# 加载训练阶段保存好的模型状态字典
PATH = './crfar_net.pth'
net.load_state_dict(torch.load(PATH))
# 利用模型对图片进行预测
output = net(images)        # 返回四组概率值
print(output)
# 共有十个类别，采用模型计算出概率最大的作为预测的类别
_, predicted = torch.max(output, 1)
print(labels)
print(predicted)
print((predicted == labels).sum().item())
# 打印预测标签结果
#print("predicted：", " ".join('%5s' % classes[labels[i]] for i in range(4)))
"""


PATH = './crfar_net.pth'
net = Net()
net.load_state_dict(torch.load(PATH))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        output = net(images)
        _, predicted = torch.max(output, 1)     # 这四组概率值，每组找出最大概率的索引
        print(predicted)            # 一批次数据打印一个，例如tensor([3, 8, 0, 8])
        total += len(labels)        # 一个批次有一个，例如tensor([3, 8, 8, 0])
        correct += (predicted == labels).sum().item()       # 每次预测每个tensor里面有几个正确的就累加。
print("acc正确率是:%f" % (correct/total))

"""
逻辑：
一批次四张图片，对应有四个预测值，这四个预测值都是十分类的概率值。
用torch.max(output, 1)可以获取每个图片预测值最大值和最大值所对应的索引。返回张量盛放4个索引,类似于tensor([6, 6, 4, 3])
用torch.max(output, 1)获取的索引与原标签labels进行对比，原标签类也是似于这重tensor([6, 6, 2, 1])
每一批次（4个数据）正确几个correct就加几
每训练一批次total就加4(其实也就是10000条样本)
最终进行相除(correct/total)
"""












