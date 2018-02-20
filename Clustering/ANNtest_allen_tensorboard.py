#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 15:07:38 2018

@author: hyunsu
"""
# for tensorflow, tensorboard implementation.

import tensorflow as tf
import numpy as np

trainX = np.loadtxt('cretrainX.csv', delimiter = ',')
trainY = np.loadtxt('cretrainY.csv', delimiter = ',')

testX = np.loadtxt('cretestX.csv', delimiter = ',')
testY = np.loadtxt('cretestY.csv', delimiter = ',')

X = tf.placeholder(tf.float32, [None, 43])
Y = tf.placeholder(tf.float32, [None, 2])

with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_normal([43, 20], stddev=0.01), name='W1')
    # ReLU function for layer 1. 20 nodes.
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_normal([20, 10], stddev=0.01), name='W2')
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_normal([10, 2], stddev=0.01), name='W3')
    # binary class(2).
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)
    tf.summary.scalar('cost', cost)

with tf.name_scope("accuracy"):
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

summ = tf.summary.merge_all()


saver = tf.train.Saver()
sess = tf.Session()

sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('./logs/LR_e-3')
# $ tensorboard --logdir=./logs
writer.add_graph(sess.graph)

batch_size = 128
total_batch = int(len(trainX) / batch_size)

for epoch in range(10000):
    for start, end in zip(range(0, len(trainX), batch_size),
        range(batch_size, len(trainX)+1, batch_size)):
        s = sess.run(summ, feed_dict={X: trainX[start:end], Y: trainY[start:end]})
        writer.add_summary(s, start*epoch)
        sess.run(optimizer, feed_dict={X: trainX[start:end], Y: trainY[start:end]})
    if (epoch % 500) == 0:
        saver.save(sess, './model/ANN.ckpt', epoch)
        print('Epoch:', '%04d' % (epoch +1))




