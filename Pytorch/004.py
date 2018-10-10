'''
@time:2018-10-09
@author:MrGao
@describe:
    在Pytorch中创建神经元
'''
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class BoWClassifier(nn.Module): # 从 nn.Module继承!
    def __init__(self,num_lables,vocab_size):
        # 在 nn.Module中调用初始化函数. 不要被这个困惑,
        # 这个做法经常在 nn.Module见到
        super(BoWClassifier, self).__init__()

        # 定义你需要的变量. 在本例中, 我们需要affine mapping的系数 A 和 b.
        # Torch 定义了可提供 affine map的nn.Linear().
        self.linear = nn.Linear(vocab_size,num_lables)

    def forward(self, bow_vec):
        # 将输入引入到线性神经元层中, 随后引入到log_softmax.
        # 在torch.nn.functional中有很多非线性和其他的函数
        return F.log_softmax(self.linear(bow_vec), dim=1)

def makeBowVector(sentence,word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    # reshape
    return vec.view(1,-1)
def makeTarget(label,label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

if __name__ == '__main__':
    data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
            ("Give it to me".split(), "ENGLISH"),
            ("No creo que sea una buena idea".split(), "SPANISH"),
            ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]
    test_data = [("Yo creo que si".split(), "SPANISH"),
                 ("it is lost on me".split(), "ENGLISH")]

    # word_to_ix 将在词汇中的单词映射为一个特征数,
    # 这个特征数就是单词在词袋中的索引
    word_to_ix = {}
    for sent,label in data+test_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    vocabSize = len(word_to_ix)
    num_labels = 2

    # model知道它的系数.第一个输出的是A, 第二个是b
    model = BoWClassifier(num_labels,vocabSize)
    for parms in model.parameters():
        print(parms)

    label_to_ix = {"SPANISH": 0, "ENGLISH": 1}
    # 在我们训练前运行测试集, 去看看前后的变化
    # 要运行该模型,请传入一个BoW vector,但要将其封装在一个autograd.Variable中.
    for instance, label in test_data:
        bow_vec = autograd.Variable(makeBowVector(instance, word_to_ix))
        log_probs = model(bow_vec)
        print(log_probs)

    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(),lr=0.1)
    # train
    for epoch in range(100):
        for instance,label in data:
            # 步骤1.牢记 Pytorch会积累梯度.我们需要在每一例前清理掉
            model.zero_grad()
            # 步骤2.制作我们的BOW向量并且我们必须将目标封装在变量中并且为整数
            bow_vec = autograd.Variable(makeBowVector(instance, word_to_ix))
            target = autograd.Variable(makeTarget(label, label_to_ix))
            # 步骤3.Run our forward pass.
            log_probs = model(bow_vec)
            # 步骤4.计算损失,梯度,通过调用optimizer.step()来更新系数
            loss = loss_function(log_probs, target)
            loss.backward()
            optimizer.step()
    # test
    for instance, label in test_data:
        bow_vec = autograd.Variable(makeBowVector(instance, word_to_ix))
        log_probs = model(bow_vec)
        print(log_probs,np.argmax(log_probs.data))

