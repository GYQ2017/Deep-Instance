'''
@time:2018-10-10
@author:MrGao
@describe:
    Pytorch中的单词嵌入Cbow模型
'''

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Cbow(nn.Module):
    def __init__(self,vocab_size,embedding_dim,context_size):
        super(Cbow, self).__init__()
        # parm1:词汇量 parm2:维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(context_size * embedding_dim*2, 128)
        self.linear2 = nn.Linear(128, vocab_size)
    def forward(self, inputs):
        embeds =  self.embedding(inputs).view((1,-1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

def make_context_vector(context,word_to_idx):
    idx = [word_to_idx[w] for w in context]
    tensor = torch.LongTensor(idx)
    return autograd.Variable(tensor)
def make_target_vector(target,word_to_idx):
    idx = [word_to_idx[target]]
    tensor = torch.LongTensor(idx)
    return autograd.Variable(tensor)

if __name__ == '__main__':
    text = """We are about to study the idea of a computational process.
    Computational processes are abstract beings that inhabit computers.
    As they evolve, processes manipulate other abstract things called data.
    The evolution of a process is directed by a pattern of rules called a program. 
    People create programs to direct processes. 
    In effect,we conjure the spirits of the computer with our spells.""".split()
    vocab = set(text)
    vocab_size = len(vocab)
    word_to_idx = {word:i for i ,word in enumerate(vocab)}

    data = []
    for i in range(2,len(text)-2):
        context = [text[i-2],text[i-1],text[i+1],text[i+2]]
        target = text[i]
        data.append((context,target))
    print(data[:2])

    lossed = []
    loss_function = nn.NLLLoss()
    model = Cbow(len(vocab),embedding_dim=10,context_size=2)
    optimizer = optim.SGD(model.parameters(), lr=0.001)

    for epoch in range(20):
        total_loss = torch.Tensor([0])
        for context,target in data:
            # 步骤1.准备好进入模型的数据
            context_var = make_context_vector(context,word_to_idx)
            # 步骤2.回调 *积累* 梯度.在进入一个实例前,需要将之前的实力梯度置零
            model.zero_grad()
            # 步骤3.运行正向传播,得到单词的概率分布
            log_probs = model(context_var)
            # 步骤4.计算损失函数.(再次注意,Torch需要将目标单词封装在变量中)
            target_var = make_target_vector(target,word_to_idx)
            loss = loss_function(log_probs, target_var)
            # 步骤5.反向传播并更新梯度
            loss.backward()
            optimizer.step()

            total_loss += loss.data
        lossed.append(total_loss)
    for i in range(20):
        print(lossed[i])