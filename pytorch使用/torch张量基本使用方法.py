from __future__ import print_function
import torch
import numpy as np

# 创建一个没有初始化的矩阵,里面是内存随便给的数据。
x = torch.empty(5, 3)
print(x)

# 创建一个有初始化的矩阵,rand 按照高斯分布进行初始化。
x = torch.rand(5, 3)
print(x)

# 创建一个全0矩阵并可指定数据元素的类型为long
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# 直接创建张量
x = torch.tensor([2.5, 3.5])
print(x)

# 创建一个新张量，张量里面全是1
x = x.new_ones(5, 3, dtype=torch.double)
# 通过已有张量，创建相同尺寸的张量
y = torch.rand_like(x, dtype=torch.float)
print(y)
print(y.size())
# 返回的内容本质上是一个元组,可以用两个值进行获取
# torch.Size([5, 3])


# 张量之间的加法操作
print(x+y)
print(torch.add(x, y))
# 创建一个空的张量（张量大小必须和加和后的张量大小相同），将将加和的值放在这个空的张量里面
result = torch.empty(5, 3)
print(torch.add(x, y, out=result))
print(y.add(x))
print(result)

# 张量切片
print(x[:, 1])  # 所有行第一列
print(x[:, :2])     # 所有行的前两列

# 改变张量的形状
x = torch.randn(4, 4)
y = x.view(16)      # 变成一维数组，数组个数必须是张量元素数量。
z = x.view(-1, 8)   # -1代表自动匹配的个数,后面的数必须是能被总数量整除的数。
print(x.size(), y.size(), z.size())

# 如果张量里面只有一个元素(只适用1个元素的张量)，item()可以将元素拿出来，变成python类型的数字
x = torch.randn(1)
print(x.item())

# 获取多个元素张量里面的数据
a = torch.randn(2, 3)
print(a[0, 0].tolist())


# torch的tensor和numpy是内存共享的
a = torch.ones(5)
b = a.numpy()
print(a, b)
a.add_(1)
print(a, b)     # 在tensor a上加了1， 那么在numpy上也加了1， 因为共享内存

# 将Numpy array转换成Torch Tensor
a = np.ones(5)
b = torch.from_numpy(a)
print(a, b)
np.add(a, 1, out=a)
print(a, b)

# 在GPU上创建张量，将CPU上的张量转移到GPU上并进行相加
# 判断服务器上是否已经安装GPU
if torch.cuda.is_available():
    # 定义设备
    device = torch.device('cuda')
    # 直接在GPU上创建张量y，在CPU上创建张量x
    y = torch.randn(3, 4, device=device)
    x = torch.ones_like(x)
    # 将CPU的x转换成GPU的x
    x = x.to(device)
    # 只有将CPU上的数据转换到GPU上，才能进行加法运算。
    # 计算的话，要么同一台CPU上，要么同一台GPU上。
    print(y + x)