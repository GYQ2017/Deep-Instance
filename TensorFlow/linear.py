'''
功能：
    现行回归，根据2*x+0.5这条直线随机生成数据
Author:GYQ
Time:20180205
'''

import numpy as np
import tensorflow as tf
import random

# parameters
epoch = 15
w = 2.0
b = 0.5
num = 1000

# 随机生成x and y
def generate_data(w,b,num):
    w = 2.0
    b = 0.5
    # 生成一个指定范围内的随机浮点数
    for i in range(num):
        x = random.uniform(1,10)
        y = w * x + b
        x = np.array([[x]])
        y = np.array([[y]])
        yield x,y
# w*x+b
if __name__ == '__main__':
    x = tf.placeholder(tf.float32,[None,1])
    y = tf.placeholder(tf.float32,[None,1])

    weight = tf.Variable(tf.zeros([1,1]))
    biases = tf.Variable(tf.zeros([1]))

    logits = tf.matmul(x,weight) + biases
    loss = tf.reduce_mean(tf.square(y-logits))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(epoch):
            for sample,label in generate_data(w,b,num=num):
                _,loss_train = sess.run([optimizer,loss],feed_dict={x:sample,y:label})
                print('w:%f, b:%f, loss:%f'%(sess.run(weight),sess.run(biases),loss_train))

# w2*x^2 + w1*x + b
if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 1])
    y = tf.placeholder(tf.float32, [None, 1])

    weight1 = tf.Variable(tf.zeros([1,1]))
    weight2 = tf.Variable(tf.zeros([1,1]))
    biases = tf.Variable(tf.zeros([1]))

    logits = tf.matmul(x*x, weight2) + tf.matmul(x, weight1) + biases
    loss = tf.reduce_mean(tf.square(y - logits))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        for i in range(epoch):
            for sample, label in generate_data(w, b, num=num):
                _, loss_train = sess.run([optimizer, loss], feed_dict={x: sample, y: label})
                print('w2:%f, w1:%f, b:%f, loss:%f' % (sess.run(weight2),sess.run(weight1), sess.run(biases), loss_train))

