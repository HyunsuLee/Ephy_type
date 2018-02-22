#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:07:38 2018

@author: hyunsu
"""
# for tensorflow, tensorboard implementation.

import tensorflow as tf
import numpy as np

trainX = np.loadtxt('cretrainX_minmax.csv', delimiter = ',')
trainY = np.loadtxt('cretrainY.csv', delimiter = ',')

testX = np.loadtxt('cretestX_minmax.csv', delimiter = ',')
testY = np.loadtxt('cretestY.csv', delimiter = ',')

X = tf.placeholder(tf.float32, [None, 43])
Y = tf.placeholder(tf.float32, [None, 2])

with tf.name_scope('layer1'):
    Xavier_init = np.sqrt(2) * np.sqrt(2 / (43 + 20))
    W1 = tf.Variable(tf.truncated_normal([43, 20], stddev=Xavier_init), name='W1')
    # ReLU function for layer 1. 20 nodes.
    b1 = tf.Variable(tf.random_normal([20]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)
    tf.summary.histogram("W1", W1)

with tf.name_scope('layer2'):
    Xavier_init = np.sqrt(2) * np.sqrt(2 / (20 + 10))
    W2 = tf.Variable(tf.truncated_normal([20, 10], stddev=Xavier_init), name='W2')
    b2 = tf.Variable(tf.random_normal([10]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)
    tf.summary.histogram("W2", W2)

with tf.name_scope('output'):
    Xavier_init = np.sqrt(2 / (10 + 2))
    W3 = tf.Variable(tf.truncated_normal([10, 2], stddev=Xavier_init), name='W3')
    b3 = tf.Variable(tf.random_normal([2]))
    # binary class(2).
    model = tf.matmul(L2, W3) + b3
    tf.summary.histogram("W3", W2)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.0001).minimize(cost)
    tf.summary.scalar('cost', cost)

with tf.name_scope("accuracy"):
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

summ = tf.summary.merge_all()


saver = tf.train.Saver()
sess = tf.Session()

sess.run(tf.global_variables_initializer())
summaries_dir = './logs/180222/LR_e-4_50000/'
train_writer = tf.summary.FileWriter(summaries_dir + '/train')
test_writer = tf.summary.FileWriter(summaries_dir + '/test')
# $ tensorboard --logdir=./logs
train_writer.add_graph(sess.graph)

# batch_size = 128
# total_batch = int(len(trainX) / batch_size)

for epoch in range(50000):
    # for start, end in zip(range(0, len(trainX), batch_size),
    #    range(batch_size, len(trainX)+1, batch_size)):
    #    sess.run(optimizer, feed_dict={X: trainX[start:end], Y: trainY[start:end]})
    sess.run(optimizer, feed_dict={X: trainX, Y: trainY})
    if (epoch % 5) == 0:
        train_acc, train_summ = sess.run([accuracy, summ], feed_dict={X: trainX, Y: trainY})
        train_writer.add_summary(train_summ, epoch)

        test_acc, test_summ = sess.run([accuracy, summ], feed_dict={X: testX, Y: testY})
        test_writer.add_summary(test_summ, epoch)

    if (epoch % 500) == 0:
        saver.save(sess, './model/180222/LR_e-4_50000/ANN.ckpt', epoch)
        print('Epoch:', '%04d' % (epoch +1))




