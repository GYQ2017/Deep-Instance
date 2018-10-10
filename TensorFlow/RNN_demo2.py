'''
demo2.py;
    基于tf中自己写数据迭代的RNN实例
Author：GYQ
Time:2018/02/01
'''

import tensorflow as tf
import numpy as np
import jieba
import os
import tflearn
import time
from tflearn.data_utils import to_categorical, pad_sequences
# from sklearn.model_selection import train_test_split
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

# 迭代
def batch_generator(samples, labels, batch_size):
    # shape函数得到行和列
    size = samples.shape[0]  # 得到行数
    indices = np.arange(size)  # 生成0-(size-1)的array
    np.random.shuffle(indices)  # 打乱顺序
    X_copy = samples[indices]
    Y_copy = labels[indices]
    i = 0
    while i + batch_size <= size:
        yield X_copy[i:i + batch_size], Y_copy[i:i + batch_size]
        i += batch_size

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
    rate = int(len(X) * test_rate)
    X_train = X[rate:]
    Y_train = Y[rate:]
    X_test = X[:rate]
    Y_test = Y[:rate]
    return X_train,Y_train,X_test,Y_test

def build_vocabulary(X_train,X_test):
    print('process vocabulary')
    vp = tflearn.data_utils.VocabularyProcessor(max_document_length=time_steps, min_frequency=0)
    X_train = np.array(list(vp.fit_transform(X_train)))
    X_test = np.array(list(vp.fit_transform(X_test)))

    X_train = pad_sequences(X_train, maxlen=time_steps, value=0.)
    X_test = pad_sequences(X_test, maxlen=time_steps, value=0.)

    n_words = len(vp.vocabulary_)
    print('total words:%d' % n_words)
    return X_train,X_test,n_words

def create_model(samples,labels):
    print('bulid model')
    # samples = tf.nn.embedding_lookup(embedding_var,samples)

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, samples, dtype=tf.float32)

    # define w and b
    weights = tf.Variable(tf.random_normal([num_hidden, num_class]))
    biases = tf.Variable(tf.random_normal([num_class]))

    logits = tf.matmul(states.h,weights)+biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits
                          (logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    corrected_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(corrected_pred, tf.float32))
    return loss,accuracy,optimizer

if __name__ == '__main__':
    samples,labels = get_data(name_sent='../sentence.txt', name_label='../label.txt')
    X_train, Y_train, X_test, Y_test = Divide_dataet(samples, labels, test_rate)
    X_train, X_test, n_words = build_vocabulary(X_train, X_test)
    Y_train = to_categorical(Y_train, nb_classes=num_class)
    Y_test = to_categorical(Y_test, nb_classes=num_class)
    # embedding layer
    embedding_var = tf.Variable(tf.random_uniform([n_words, embedding_dim], -1.0, 1.0), trainable=True)

    batch_ph = tf.placeholder(tf.int32, [None, time_steps])
    target_ph = tf.placeholder(tf.int32, [None, num_class])
    batch_ph_embedd = tf.nn.embedding_lookup(embedding_var, batch_ph)
    # target_ph = tf.one_hot(target_ph,num_class)
    loss_op,acc_op,optimizer_op = create_model(batch_ph_embedd,target_ph)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print('train')
        time_s = time.time()
        step = 0
        for epoch in range(num_epoch):
            for sample_batch,label_batch in batch_generator(X_train,Y_train,batch_size=batch_size):
                # print(sample_batch.shape,label_batch.shape)
                _,loss_train,acc_train = sess.run([optimizer_op,loss_op,acc_op],
                                                  feed_dict={batch_ph:sample_batch,
                                                             target_ph:label_batch})
                if (step+1)%50 == 0:
                    print('step:%d,loss:%f,acc:%f'%(step+1,loss_train,acc_train))
                step += 1
        print('train time:%s'%(time.time()-time_s))

        print('test')
        accs = []
        for sample_test_batch,label_test_batch in batch_generator(X_test,Y_test,batch_size):
            acc_test = sess.run(acc_op,
                                feed_dict={batch_ph:sample_test_batch,
                                           target_ph:label_test_batch})
            accs.append(acc_test)
        test_acc = sum(accs)*1.0 / len(accs)
        print('total time:%s,test_acc=%f'%(time.time()-time_s,test_acc))
        '''
            # testing
            for step in range(1, training_steps + 1):
                batch_x1, batch_y1 = next(test_batch_generator)

                loss1_test, acc_test = sess.run([loss, accuracy],
                                                feed_dict={x: batch_x1, y: batch_y1})
                accuracy_test += acc_test
                loss_test += loss1_test
            accuracy_test /= training_steps
            loss_test /= training_steps

            print('............................', epoch, 'epoch  end......................................')
            print("epoch: {} ,train_loss: {:.3f}, test_loss: {:.3f}, train_acc: {:.3f}, test_acc: {:.3f}".format(
                epoch, loss_train, loss_test, accuracy_train, accuracy_test))
            time_e = time.time()
            print('epoch :', epoch, ' total time :', time_e - time_s)
'''
