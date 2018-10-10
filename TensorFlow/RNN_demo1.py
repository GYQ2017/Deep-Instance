'''
demo1.py;
    基于tf中Dataset输入的RNN实例
Author:GYQ
Time:2018/02/01
'''

from tflearn.data_utils import to_categorical, pad_sequences
# from sklearn.model_selection import train_test_split
import tensorflow as tf
import time
import pickle
import jieba
import tflearn
import numpy as np

# Paramenters
learning_rate = 0.01
batch_size = 32
time_steps = 50

embedding_dim = 300
num_epoch = 20
num_hidden = 64
num_class = 2
test_rate = 0.2

# 获取数据，返回列表
def get_data(sample_name,label_name):
    print('Get Data')
    # 使用自定义词典
    jieba.load_userdict('vocabulary.txt')
    X = []
    Y = []
    with open(sample_name,'rb') as fo1:
        list_x = pickle.load(fo1)
        for i in list_x:
            i = jieba.cut(i)
            X.append(' '.join(i))
    with open(label_name,'rb') as fo2:
        Y = pickle.load(fo2)
    return X,Y

# 建立词典，为embedding做准备
def build_vocabulary(X):
    # 生成词典
    vp = tflearn.data_utils.VocabularyProcessor(max_document_length=time_steps, min_frequency=0)
    # 返回单词索引
    X = np.array(list(vp.fit_transform(X)))
    # 该实例循环神经网络支持不定长句子
    # X = pad_sequences(X, maxlen=time_steps, value=0.)
    n_words = len(vp.vocabulary_)
    print('total words:%d' % n_words)
    return X, n_words

# 分割数据集，返回训练集和测试集
def Divide_dataset(X,Y,test_rate):
    # X_train, Y_train, X_test, Y_test = train_test_split(X,Y,test_size=test_rate,random_state=66)
    rate = int(len(X) * test_rate)
    X_train = X[rate:]
    Y_train = Y[rate:]
    X_test = X[:rate]
    Y_test = Y[:rate]
    return X_train, Y_train, X_test, Y_test

# 建立模型函数
def create_model(samples,labels):
    print('bulid model')
    # 定义LSTM神经网络
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, inputs=samples, dtype=tf.float32)
    # define w and b
    weights = tf.Variable(tf.random_normal([num_hidden, num_class]))
    biases = tf.Variable(tf.random_normal([num_class]))
    logits = tf.matmul(states.h, weights) + biases
    # define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                          (logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    corrected_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(corrected_pred, tf.float32))
    return loss, accuracy, optimizer

def parse(samples,labels):
    X_sents = tf.nn.embedding_lookup(embedding_var, samples)
    Y_labels = tf.one_hot(labels, num_class)
    return X_sents, Y_labels

if __name__ == '__main__':
    samples,labels = get_data(sample_name='sentence.txt',label_name='label.txt')
    samples,n_words = build_vocabulary(samples)
    samples_train,labels_train,samples_test,labels_test = \
        Divide_dataset(samples,labels,test_rate=test_rate)
    # embedding layer
    embedding_var = tf.Variable(tf.random_uniform([n_words,embedding_dim],-1.0,1.0),trainable=True)

    # 使用tf API自带的输入DataSet，建立数据迭代
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(samples_train, dtype=tf.int32),
         tf.convert_to_tensor(labels_train, dtype=tf.int32)))
    test_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.convert_to_tensor(samples_test, dtype=tf.int32),
         tf.convert_to_tensor(labels_test, dtype=tf.int32)))
    train_dataset = train_dataset.shuffle(len(samples_train)).map(parse).\
        batch(batch_size).repeat(num_epoch)
    test_dataset = test_dataset.shuffle(len(samples_test)).map(parse).\
        batch(batch_size)
    iterator = tf.data.Iterator.from_structure(output_types=train_dataset.output_types,
                                               output_shapes=train_dataset.output_shapes)
    train_data_init = iterator.make_initializer(train_dataset)
    test_data_init = iterator.make_initializer(test_dataset)
    sample_batch, label_batch = iterator.get_next()

    loss_op,accuracy_op,optimizer_op = create_model(sample_batch,label_batch)
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        print('training')
        sess.run(train_data_init)
        time_s = time.time()
        try:
            step = 0
            while True:
                _, loss_train, acc_train = sess.run([optimizer_op, loss_op, accuracy_op])
                if (step+1) % 50 == 0:
                    print('step:%d,loss:%f,acc:%f'%(step+1,loss_train,acc_train))
                step += 1
        except tf.errors.OutOfRangeError:
            print('train data finish')
        finally:
            print('time:%s'%(time.time()-time_s))

        print('testing')
        accs = []
        sess.run(test_data_init)
        try:
            while True:
                test_acc = sess.run(accuracy_op)
                accs.append(test_acc)
        except tf.errors.OutOfRangeError:
            print('test data finish')
        finally:
            test_acc = sum(accs)*1.0/len(accs)
            print('total time:%s,test_acc=%f'%(time.time()-time_s,test_acc))


