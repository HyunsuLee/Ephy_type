#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
created 2018-03-01
@author: hyunsu
"""
# binary model 180301.py tensorboard evaluated(results from 1st fine tuning) 
# LR3.792e-04Beta1.762e-05 is best

import tensorflow as tf
import numpy as np
import random
import os

data_path = './180228tensordata_minmax/'
log_path = '/180301binary_full/final/'
model_dir = './model/' + log_path # for model saver

testX = np.loadtxt(data_path + 'BtestX_minmax.csv', delimiter = ',')
testY = np.loadtxt(data_path + 'BtestY.csv', delimiter = ',')

X = tf.placeholder(tf.float32, [None, 43]) # for full model, 43 input features
Y = tf.placeholder(tf.float32, [None, 2]) # binary E vs I class
keep_prob = tf.placeholder(tf.float32)
is_training_holder = tf.placeholder(tf.bool)
# learning_rate = tf.placeholder(tf.float32)
L2beta = tf.placeholder(tf.float32)
epsilon = 1e-3 # for Batch normalization
layer1_shape = [43, 20]
layer2_shape = [20, 10]
output_shape = [10, 2]


def weight_init(shape, name_for_weight):
    Xavier_init = np.sqrt(2.0) * np.sqrt(2.0 / np.array(shape).sum())
    weights = tf.truncated_normal(shape, stddev = Xavier_init)
    return tf.Variable(weights, name = name_for_weight)

with tf.name_scope('layer1'):
    W1 = weight_init(layer1_shape, 'W1')
    z1 = tf.matmul(X, W1)
    BN1 = tf.contrib.layers.batch_norm(z1, center = True, scale = True,
          is_training = is_training_holder) 
    L1 = tf.nn.relu(BN1)
    L1 = tf.nn.dropout(L1, keep_prob)
    
with tf.name_scope('layer2'):
    W2 = weight_init(layer2_shape, 'W2')
    z2 = tf.matmul(L1, W2)
    BN2 = tf.contrib.layers.batch_norm(z2, center = True, scale = True,
          is_training = is_training_holder) 
    L2 = tf.nn.relu(BN2)
    L2 = tf.nn.dropout(L2, keep_prob)

with tf.name_scope('output'):
    W3 = weight_init(output_shape, 'W3')
    b3 = tf.Variable(tf.random_normal([output_shape[1]]))
    model = tf.matmul(L2, W3) + b3

model_checkpoint_all = os.listdir(model_dir)

models_list = []
for model_checkpoints in model_checkpoint_all:
    if model_checkpoints[-5:] == '.meta':
        models_list += [model_checkpoints[:-5]]

sess = tf.InteractiveSession()
saver = tf.train.Saver(tf.global_variables())

random_L2beta = 0.000017 

lowest_cost = 1.0
for model_checks in models_list:
    saver = tf.train.import_meta_graph(model_dir + model_checks + '.meta')
    saver.restore(sess, model_dir + model_checks)

    with tf.name_scope("accuracy"):
        is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    
    with tf.name_scope('optimizer'):
        base_cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
        lossL2 =  tf.reduce_mean(tf.nn.l2_loss(W1) + 
            tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3)) * L2beta
        cost = base_cost + lossL2 
    
    cost = sess.run(cost, 
                feed_dict={X: testX, Y: testY, keep_prob: 1.0,
                is_training_holder: 0, L2beta: random_L2beta})
    accuracy = sess.run(accuracy, 
                feed_dict={X: testX, Y: testY, keep_prob: 1.0, 
                is_training_holder: 0, L2beta: random_L2beta})
    print('Model:', model_checks, 'Test accuracy:', '{:.4f}'.format(accuracy),
        'Test cost:', '{:.4f}'.format(cost))
    
    if cost < lowest_cost:
        lowest_cost = cost
        best_model = model_checks

print('Lowest cost:', '{:.4f}'.format(lowest_cost), 'Best model:', best_model)

saver = tf.train.import_meta_graph(model_dir + best_model + '.meta')
saver.restore(sess, model_dir + best_model)

model_eval = sess.run(model, feed_dict = {X: testX, Y: testY, keep_prob: 1.0,
                is_training_holder: 0, L2beta: random_L2beta})
model_prob = sess.run(tf.nn.softmax(model_eval))

np.savetxt('./results/BtestY_ANNmodel_prob.csv', model_prob, delimiter=',')
np.savetxt('./results/BtestY.csv', testY, delimiter = ',')
