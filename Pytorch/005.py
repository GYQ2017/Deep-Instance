'''
@time:2018-10-10
@author:MrGao
@describe:
    Pytorch中的单词嵌入,即N-Gram语言模型(N=2)
'''

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

CONTEXT_SIZE = 2
EMBEDDING_DIM = 10

class NGramLanguageModel(nn.Module):
    def __init__(self,vocab_size,embedding_dim,context_size):
        super(NGramLanguageModel, self).__init__()
        # parm1:词汇量 parm2:维度
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.linear1 = nn.Linear(context_size*embedding_dim,128)
        self.linear2 = nn.Linear(128,vocab_size)
    def forward(self, inputs):
        embeds = self.embedding(inputs).view((1,-1))
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out,dim=1)
        return log_probs

if __name__ == '__main__':
    sentence = """When forty winters shall besiege thy brow,
    And dig deep trenches in thy beauty's field,
    Thy youth's proud livery so gazed on now,
    Will be a totter'd weed of small worth held:
    Then being asked, where all thy beauty lies,
    Where all the treasure of thy lusty days;
    To say, within thine own deep sunken eyes,
    Were an all-eating shame, and thriftless praise.
    How much more praise deserv'd thy beauty's use,
    If thou couldst answer 'This fair child of mine
    Shall sum my count, and make my old excuse,'
    Proving his beauty by succession thine!
    This were to be new made when thou art old,
    And see thy blood warm when thou feel'st it cold.""".split()
    # 建造一系列元组.每个元组([word_i-2,word_i-1 ],目标单词)
    trigrams = [([sentence[i],sentence[i+1]],sentence[i+2])for i in
                range(len(sentence)-2)]
    print(trigrams[:2])
    vocab = set(sentence)
    word_to_ix = {word:i for i,word in enumerate(vocab)}

    lossed = []
    loss_function = nn.NLLLoss()
    model = NGramLanguageModel(len(vocab),EMBEDDING_DIM,CONTEXT_SIZE)
    optimizer = optim.SGD(model.parameters(),lr=0.001)

    # train
    for epoch in range(10):
        total_loss = torch.Tensor([0])
        for context,target in trigrams:
            # 步骤1.准备好进入模型的数据
            context_idx = [word_to_ix[w] for w in context]
            context_var = autograd.Variable(torch.LongTensor(context_idx))
            # 步骤2.回调 *积累* 梯度.在进入一个实例前,需要将之前的实力梯度置零
            model.zero_grad()
            # 步骤3.运行正向传播,得到单词的概率分布
            log_probs = model(context_var)
            # 步骤4.计算损失函数.(再次注意,Torch需要将目标单词封装在变量中)
            target_var = autograd.Variable(torch.LongTensor([word_to_ix[target]]))
            loss = loss_function(log_probs,target_var)
            # 步骤5.反向传播并更新梯度
            loss.backward()
            optimizer.step()

            total_loss += loss.data
        lossed.append(total_loss)
    print(lossed)