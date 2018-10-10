'''
@time:2018-10-09
@author:MrGao
@describe:
    Pytorch 深度学习构建模块:映射
'''
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


x = torch.Tensor([[1.,-2.,3.],[4.,5.,6.]])
data = autograd.Variable(x)
# 线性Linear
linear = nn.Linear(3,1)
print(linear(data))

# non-linearites不拥有参数.意味着它们再训练时没有可以更新的参数
print(data)
print(F.relu(data)) # relu(x)=max(0,x)
