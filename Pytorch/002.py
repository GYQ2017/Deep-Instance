'''
@time:2018-10-09
@author:MrGao
@describe:
    Pytorch 计算图和自动求导
'''
import torch
import torch.autograd as autograd

# 这些是Tensor类型,反向是不可能的
x = torch.Tensor([1., 2., 3.])
y = torch.Tensor([4., 5., 6.])

var_x = autograd.Variable(torch.Tensor([1., 2., 3.]), requires_grad=True)
var_y = autograd.Variable(torch.Tensor([4., 5., 6.]), requires_grad=True)
# var_z 包含了足够的信息去计算梯度,如下所示
var_z = var_x + var_y
print(var_z)
var_s = var_z.sum()
print(var_s)
# backward 必须是标量
var_s.backward()
print(var_x.grad)