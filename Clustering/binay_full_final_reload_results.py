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

data_path = './180228tensordata_minmax/'
log_path = '/180301binary_full/final/'
summaries_dir = './logs/' + log_path # for tensorboard summary
model_dir = './model/' + log_path # for model saver

# saver restore 후에 
'''
model_eval = sess.run(model, feed_dict = {X: testX})
model_prob = sess.run(tf.nn.softmax(model_eval))
np.savetxt('model_prob.csv', model_prob, delimiter=',')
'''
# 로 results folder에 넣을것. 
