#!/usr/bin/evn python
#-*- coding: utf-8 -*-

# ===================================
# Filename : mylstm.py
# Author : GT
# Create date : 17-09-22 12:15:13
# Description:
# ===================================


# Script starts from here

# this is for chinese characters
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')


import numpy as np
import nltk
import tensorflow as tf
import languageModel 


def getData():
    gd = languageModel.getData()
    x, y = gd.encode('./DateSet/reddit-comments-2015-08.csv')
    return x, y

class RNN_model(object):
    def __init__(self, config):
        self.keep_prob = config.keep_prob
        self.x = tf.placeholder(tf.int32, shape = (config.batch_size, config.setp_num), name = 'inputs')
        self.y = tf.placeholder(tf.int32, shape = (config.batch_size, config.setp_num), name = 'targets')
        lstm = tf.contrib.rnn.LSTMCell(config.hidden_size)
        if self.keep_prob < 1:
            lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob = self.keep_prob)
        cells = tf.contrib.rnn.MultiRNNCell([lstm])
        self.initial_state = cells.zero_state(config.batch_size, tf.float32)
        inputs = tf.one_hot(self.x, config.vocab_size)
        outputs, stats = tf.nn.dynamic_rnn(cells, inputs, initial_state = self.initial_state)
        self.final_state = stats
        outputs = tf.reshape(outputs, (-1, config.hidden_size))
        with tf.variable_scope('softmax'):
            softmax_w = tf.Variable(tf.truncated_normal([config.hidden_size, config.vocab_size], stddev = 0.1))
            softmax_b = tf.Variable(tf.zeros(config.vocab_size))
        logits = tf.matmul(outputs, softmax_w) + softmax_b
        self.out = tf.nn.softmax(logits)
        _y = tf.one_hot(self.y, config.vocab_size)
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = _y)
        self.loss = tf.reduce_mean(self.loss)
        self.train = tf.train.AdamOptimizer(config.rate).minimize(self.loss)

class CONFIG(object):
    keep_prob = 0.5
    batch_size = 1
    hidden_size = 512
    vocab_size = 8000
    setp_num = None
    rate = 0.001

if __name__ =='__main__':
    cfg = CONFIG()
    model = RNN_model(cfg)
    x, y = getData()
    
    print np.array(x[0]).shape 
    sess = tf.InteractiveSession() 
    sess.run(tf.initialize_all_variables())

    new_state = sess.run(model.initial_state)
    for i in range(100):
        for j in range(100):
            feed = {model.x:[np.array(x[j])],model.y:[np.array(y[j])]}
            new_state ,loss, _= sess.run([model.final_state, model.loss, model.train],feed_dict = feed)
        print 'epcoh: ',i, ' loss:', loss     
    
