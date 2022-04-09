import torch

x = torch.ones(3, 3, requires_grad=True)
y = x + 1
print(x.grad_fn)
print(y.grad_fn)
"""
x是自定义的tensor
None
y是延伸出来的tensor
<AddBackward0 object at 0x000001BF7307B3A0>
"""

z = y * y * 3
out = z.mean()
print(z, out)

"""
tensor([[12., 12., 12.],
        [12., 12., 12.],
        [12., 12., 12.]], grad_fn=<MulBackward0>) tensor(12., grad_fn=<MeanBackward0>)
"""

a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.grad_fn)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b)
print(b.grad_fn)

# 设置反向传播
out.backward()
print(x.grad)       # grad保存了tensor反向传播的梯度
print(x.requires_grad)
with torch.no_grad():   # 不对他进行梯度求导
    print((x**2).requires_grad)




