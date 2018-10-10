'''
@time:2018-10-10
@author:MrGao
@describe:
    Pytorch中的LSTM
'''
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class LSTMTagger(nn.Module):
    def __init__(self,embedding_dim,hidden_dim,vocab_size,tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size,embedding_dim)
        # LSTM以word_embeddings作为输入,输出维度为hidden_dim的隐状态值
        self.lstm = nn.LSTM(embedding_dim,hidden_dim)

        # 线性层将隐状态空间映射到标注空间
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
    def init_hidden(self):
        # 开始时刻, 没有隐状态
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out,self.hidden = self.lstm(
            embeds.view(len(sentence),1,-1),self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence),-1))
        tag_score = F.log_softmax(tag_space,dim=1)
        return tag_score

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

if __name__ == '__main__':
    data = [
        ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
        ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
    ]
    word_to_idx ={}
    for sent,tags in data:
        for word in sent:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
    tag_to_idx = {'DET':0,'NN':1,'V':2}
    vocab_size = len(word_to_idx)

    model = LSTMTagger(EMBEDDING_DIM,HIDDEN_DIM,vocab_size,len(tag_to_idx))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(),lr = 0.1)

    # train
    for epoch in range(300):
        for sentence,tags in data:
            # step1
            model.zero_grad()
            # 此外还需要清空 LSTM 的隐状态,将其从上个实例的历史中分离出来
            model.hidden = model.init_hidden()
            # step2
            sentence_in = prepare_sequence(sentence,word_to_idx)
            targets = prepare_sequence(tags,tag_to_idx)
            # step3
            tag_score = model(sentence_in)
            # step4
            loss = loss_function(tag_score,targets)
            loss.backward()
            optimizer.step()
    # 查看训练后得分的值
    inputs = prepare_sequence(data[0][0], word_to_idx)
    tag_scores = model(inputs)
    print(tag_scores.data)
    print(torch.argmax(tag_scores.data,1))

