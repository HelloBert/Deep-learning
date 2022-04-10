import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 定义网络类
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 定义第一层卷积层，输入维度是1，输出维度是6，卷积核大小3*3
        self.conv1 = nn.Conv2d(1, 6, kernel_size=3)
        # 定义第二层卷积层，输入维度是6，输出维度16，卷积核大小3*3
        self.conv2 = nn.Conv2d(6, 16, kernel_size=3)
        # 定义第三层全连接神经网络
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    # 前向传播，这里输入的x必须是4维
    def forward(self, x):
        # 任意卷积层后面要接激活层和池化层,这两层不需要在初始化函数里面定义出来
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # 经过卷积池化层之后，张量要进入全连接层,进入前要调整张量的形状
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for i in size:
            num_features *= i
        return num_features

net = Net()
print(net)

# 用梯度下降法更新网络参数
optimizer = optim.SGD(net.parameters(), lr=0.01)

params = list(net.parameters())
print(len(params))
print(params[0].size())
# 一个图像，一个通道，宽和高都是32
input = torch.randn(1, 1, 32, 32)
# pytorch的forward函数会自动调用，不用手动调用
out = net(input)
print(out)


# 定义损失函数
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()
loss = criterion(out, target)
print(loss)
"""
print(loss.grad_fn)
print(loss.grad_fn.next_functions[0][0])
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])
"""

# 反向传播求梯度
# pytorch中反向求梯度首先要执行梯度清零的操作,因为梯度会被累加到grad属性里面，所以我们每次在反向传播求梯度的时候都要梯度清零。
optimizer.zero_grad()


# 在pytorch中实现一次反向传播
loss.backward()

# 更新参数
optimizer.step()

"""
# 将优化器梯度清零
optimizer.zero_grad()

# 执行网络计算并计算损失值
output = net(input)
loss = criterion(output, input)

# 执行反向传播
loss.backward()
"""

