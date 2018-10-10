'''
功能：
    使用BiGRU进行论文句子语病判断
    数据迭代方式：tf.Dataset https://tensorflow.google.cn/api_docs/python/tf/data/Dataset
步骤：
    获取句子和标签
    划分训练集和测试集
    建立词典、词向量
    训练、测试
Author：GYQ
Time:2018/01/31
'''
import tensorflow as tf
import numpy as np
import jieba
import tflearn
import time
from tflearn.data_utils import to_categorical, pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.contrib.rnn import GRUCell
import pickle

# Train Paramenters
learning_rate = 0.01
batch_size = 32
embedding_dim = 300
test_rate = 0.2
# Network Paramenters
num_epoch = 20
time_steps = 50
num_hidden = 64
num_class = 2

def get_data(name_sent,name_label):
    print('Get Data')
    X = []
    Y = []
    with open(name_sent,'rb') as fo1:
        list_x = pickle.load(fo1)
        for i in list_x:
            i = jieba.cut(i)
            X.append(' '.join(i))
    with open(name_label,'rb') as fo2:
        Y = pickle.load(fo2)
    return X,Y

def Divide_dataet(X,Y,test_rate):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_rate, random_state=123)
    return X_train,Y_train,X_test,Y_test

def build_vocabulary(X_train,X_test):
    # 生成词典
    vp = tflearn.data_utils.VocabularyProcessor(max_document_length=time_steps, min_frequency=0)
    # 返回单词索引
    X_train = np.array(list(vp.fit_transform(X_train)))
    X_test = np.array(list(vp.fit_transform(X_test)))
    X_train = pad_sequences(X_train, maxlen=time_steps, value=0.)
    X_test = pad_sequences(X_test, maxlen=time_steps, value=0.)
    n_words = len(vp.vocabulary_)
    print('total words:%d' % n_words)
    return X_train,X_test,n_words

def model(samples,labels):
    print('bulid model')
    # outputs用于Attention机制
    outpus,states = tf.nn.bidirectional_dynamic_rnn(GRUCell(num_hidden),GRUCell(num_hidden),
                                                    inputs=samples,dtype=tf.float32)
    # outpus = tf.concat(outpus,2)
    state_h = tf.concat((states[0],states[1]),1)
    print(states[0],states[1])
    print(state_h)
    # define w and b
    weights = tf.Variable(tf.random_normal([2*num_hidden, num_class]))
    biases = tf.Variable(tf.random_normal([num_class]))
    logits = tf.matmul(state_h, weights) + biases   # outputs:[batch_size,time_steps,num_hidden]
    # define loss and optimizer
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                              (logits=logits, labels=labels))
        tf.summary.scalar('loss',loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    # evaluate model
    corrected_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(corrected_pred, tf.float32))
    return loss,accuracy,optimizer

def parse(samples,labels):
    X_sents = tf.nn.embedding_lookup(embedding_var,samples)
    # X_sents = tf.reshape(X_sents,shape=(-1,))
    Y_labels = tf.one_hot(labels,num_class)
    return X_sents,Y_labels

if __name__ == '__main__':
    jieba.load_userdict('../vocabulary_02.txt')
    X,Y = get_data(name_sent='../sentence.txt',name_label='../label.txt')
    X_train,Y_train,X_test,Y_test = Divide_dataet(X,Y,test_rate)
    X_train,X_test,n_words = build_vocabulary(X_train,X_test)
    # embedding layer
    embedding_var = tf.Variable(tf.random_uniform([n_words, embedding_dim], -1.0, 1.0), trainable=True)

    # 建立Dataset，数据迭代
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(X_train,dtype=tf.int32),
         tf.convert_to_tensor(Y_train,dtype=tf.int32)))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(X_test,dtype=tf.int32),
         tf.convert_to_tensor(Y_test,dtype=tf.int32)))
    train_dataset = train_dataset.shuffle(len(X_train)).map(parse).batch(batch_size).repeat(num_epoch)
    test_dataset = test_dataset.shuffle(len(X_test)).map(parse).batch(batch_size)
    iterator = tf.data.Iterator.from_structure(output_types=train_dataset.output_types, output_shapes=train_dataset.output_shapes)
    train_data_init = iterator.make_initializer(train_dataset)
    test_data_init = iterator.make_initializer(test_dataset)
    sample_batch,label_batch = iterator.get_next()
    loss,accuracy,optimizer = model(sample_batch,label_batch)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('logs/',sess.graph)
        sess.run(init)
        print('start running')
        sess.run(train_data_init)
        time_s = time.time()
        try:
            step = 0
            while True:
                _,loss_train,acc_train = sess.run([optimizer,loss,accuracy])
                if (step+1)%50 == 0:
                    result = sess.run(merged)
                    writer.add_summary(result,step)
                    print('step:%d ,loss:%f ,acc:%f'%(step+1,loss_train,acc_train))
                step += 1
        except tf.errors.OutOfRangeError:
            print('finish')
        finally:
            print('train time:%s'%(time.time()-time_s))

        print('start testing')
        accs = []
        sess.run(test_data_init)
        try:
            while True:
                accs.append(sess.run(accuracy))
        except tf.errors.OutOfRangeError:
            print('finish')
        finally:
            print('test accuracy=%f'%(sum(accs)*1.0/len(accs)))






