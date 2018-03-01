#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: hyunsu
"""
# for tensorflow, tensorboard implementation.

import tensorflow as tf
import numpy as np

data_path = './180228tensordata_minmax/'
"""
there are total 48 csv files.

created by data_processing_180227.ipynb
    
    3 different output classification task.
    start with "B" means for binary classification(E vs I => outnode 2)
    "E" stands for excitatory transgenic line classification(outnode 10)
    "I" stands for inhibitory transgenic line classificiation(outnode 8)
    
    4 different input features.
    full model(all electrophysiology features)
    _long.csv - long square pulse protocol
    _short.csv - short square pulse protocol
    _ramp.csv - ramp pulse protocol
3 X 4 = 12. 12 different ANN models will be created.
"""
log_path = '/180228binary_full/BN_tflayer_test/'
summaries_dir = './logs/' + log_path # for tensorboard summary
model_dir = './model/' + log_path # for model saver


trainX = np.loadtxt(data_path + 'BtrainX_minmax.csv', delimiter = ',')
trainY = np.loadtxt(data_path + 'BtrainY.csv', delimiter = ',')

testX = np.loadtxt(data_path + 'BtestX_minmax.csv', delimiter = ',')
testY = np.loadtxt(data_path + 'BtestY.csv', delimiter = ',')

X = tf.placeholder(tf.float32, [None, 43]) # for full model, 43 input features
Y = tf.placeholder(tf.float32, [None, 2]) # binary E vs I class
keep_prob = tf.placeholder(tf.float32)
is_training_holder = tf.placeholder(tf.bool)
learning_rate = tf.placeholder(tf.float32)
epsilon = 1e-3 # for Batch normalization
layer1_shape = [43, 20]
layer2_shape = [20, 10]
output_shape = [10, 2]


def weight_init(shape, name_for_weight):
    Xavier_init = np.sqrt(2.0) * np.sqrt(2.0 / np.array(shape).sum())
    weights = tf.truncated_normal(shape, stddev = Xavier_init)
    return tf.Variable(weights, name = name_for_weight)

'''
def batch_norm_wrapper(inputs, is_training, decay = 0.999):

    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training == 1:
        batch_mean, batch_var = tf.nn.moments(inputs,[0])
        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, epsilon)
'''

with tf.name_scope('layer1'):
    W1 = weight_init(layer1_shape, 'W1')
    z1 = tf.matmul(X, W1)
    # BN1 = batch_norm_wrapper(z1, is_training)
    BN1 = tf.contrib.layers.batch_norm(z1, center = True, scale = True
          is_training = is_training_holder) 
    L1 = tf.nn.relu(BN1)
    L1 = tf.nn.dropout(L1, keep_prob)
    tf.summary.histogram('W1', W1)

with tf.name_scope('layer2'):
    W2 = weight_init(layer2_shape, 'W2')
    z2 = tf.matmul(L1, W2)
    # BN2 = batch_norm_wrapper(z2, is_training)
    BN2 = tf.contrib.layers.batch_norm(z2, center = True, scale = True
          is_training = is_training_holder) 
    L2 = tf.nn.relu(BN2)
    L2 = tf.nn.dropout(L2, keep_prob)
    tf.summary.histogram('W2', W2)

with tf.name_scope('output'):
    W3 = weight_init(output_shape, 'W3')
    b3 = tf.Variable(tf.random_normal([output_shape[1]]))
    model = tf.matmul(L2, W3) + b3
    tf.summary.histogram('W3', W3)


# TODO, model wrapper for trying multiple random hyperparameter.

with tf.name_scope('optimizer'):
    base_cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
    # lossL2 =  tf.reduce_mean(tf.nn.l2_loss(W1) + 
    # tf.nn.l2_loss(W2) + tf.nn.l2_loss(W3))* 0.01
    cost = base_cost # + lossL2 
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    tf.summary.scalar('cost', cost)

with tf.name_scope("accuracy"):
    is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

summ = tf.summary.merge_all()

saver = tf.train.Saver()

sess = tf.Session()

sess.run(tf.global_variables_initializer())

train_writer = tf.summary.FileWriter(summaries_dir + '/train')
test_writer = tf.summary.FileWriter(summaries_dir + '/test')
# $ tensorboard --logdir=./logs
train_writer.add_graph(sess.graph)

# batch_size = 128
# total_batch = int(len(trainX) / batch_size)

random_learning_rate = 0.001
for epoch in range(5000):
    # for start, end in zip(range(0, len(trainX), batch_size),
    #    range(batch_size, len(trainX)+1, batch_size)):
    #    sess.run(optimizer, feed_dict={X: trainX[start:end], Y: trainY[start:end]})
    sess.run(optimizer, feed_dict={X: trainX, Y: trainY, keep_prob: 0.5, 
                        is_training_holder: 1, learning_rate: random_learning_rate})
    if (epoch % 5) == 0:
        train_acc, train_summ = sess.run([accuracy, summ], 
                        feed_dict={X: trainX, Y: trainY, keep_prob: 1.0, 
                        is_training_holder: 1, learning_rate: random_learning_rate})
        train_writer.add_summary(train_summ, epoch)

        test_acc, test_summ = sess.run([accuracy, summ], 
                        feed_dict={X: testX, Y: testY, keep_prob: 1.0, 
                        is_training_holder: 0, learning_rate: random_learning_rate})
        test_writer.add_summary(test_summ, epoch)

    if (epoch % 500) == 0:
        saver.save(sess, model_dir + '/ANN.ckpt', epoch)
        print('Epoch:', '%04d' % (epoch +1))
        test_cost = sess.run(cost, 
                        feed_dict={X: testX, Y: testY, keep_prob: 1.0, 
                        is_training_holder: 0, learning_rate: random_learning_rate})
        print('Test cost:', '{:.3f}'.format(test_cost))
        print('Test accuracy:', '{:.3f}'.format(test_acc))





