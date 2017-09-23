#!/usr/bin/evn python
#-*- coding: utf-8 -*-

# ===================================
# Filename : languageModel.py
# Author : GT
# Create date : 17-09-20 18:33:43
# Description:
# ===================================


# Script starts from here

# this is for chinese characters
# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')


import numpy as np
import nltk
import csv
import itertools


def softmax(x):
    xt = np.exp(x - np.max(x))
    return xt / np.sum(xt)

class getData(object):
    def __init__(self):
        self.vocabulary_size = 8000
        self.unknown_token = 'UNKNOWN_TOKEN'
        self.sentence_start_token = 'SENTENCE_START'
        self.sentence_end_token = "SENTENCE_END"

    def encode(self, path):
        print 'Reading csv file %s' % path
        with open(path, 'rb') as f:
            reader = csv.reader(f, skipinitialspace = True)
            reader.next()
            self.sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
            self.sentences = ['%s %s %s' % (self.sentence_start_token, x, self.sentence_end_token) for x in self.sentences]
        print 'Parsed %d sentences' % len(self.sentences)
        self.tokenize_sentences = [nltk.word_tokenize(x) for x in self.sentences]
        self.word_freq = nltk.FreqDist(itertools.chain(*self.tokenize_sentences))
        print 'Found %d unique words tokens.' % len(self.word_freq)
        vocab = self.word_freq.most_common(self.vocabulary_size - 1)
        self.index_to_word = [x[0] for x in vocab]
        self.index_to_word.append(self.unknown_token)
        self.word_to_index = dict([(w, i) for i, w in enumerate(self.index_to_word)])
        print 'Using vocabulary size %d .' % self.vocabulary_size
        print 'The least frequent word in our vocabulary is %s and appear %d times.' % (vocab[-1][0], vocab[-1][1])
        for i, sent in enumerate(self.tokenize_sentences):
            self.tokenize_sentences[i] = [w if w in self.word_to_index else self.unknown_token for w in sent]
        print '\nExample sentence: %s' % self.sentences[0]
        print '\nExample sentence after encoding: %s' % self.tokenize_sentences[0]
        self.x_train = np.asarray([[self.word_to_index[w] for w in sent[:-1]] for sent in self.tokenize_sentences])
        self.y_train = np.asarray([[self.word_to_index[w] for w in sent[1:]] for sent in self.tokenize_sentences])
        return self.x_train, self.y_train


class RNN(object):
    def __init__(self, vocabulary_size, hidden_size, bptt_turns):
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.bptt_turns = bptt_turns
        self.U = np.random.uniform(-np.sqrt(1. / vocabulary_size), np.sqrt(1. / vocabulary_size), (hidden_size, vocabulary_size))
        self.V = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (vocabulary_size, hidden_size))
        self.W = np.random.uniform(-np.sqrt(1. / hidden_size), np.sqrt(1. / hidden_size), (hidden_size, hidden_size))
        
    def forward(self, x):
        C_size = len(x)
        s = np.zeros((C_size + 1, self.hidden_size))
        s[-1] = np.zeros(self.hidden_size)
        o = np.zeros((C_size, self.vocabulary_size))
        for i in np.arange(C_size):
            s[i] = np.tanh(self.U[:, x[i]] + self.W.dot(s[i - 1]))
            o[i] = softmax(self.V.dot(s[i]))
        return s, o

    def predict(self):
        maxIndex = np.argmax(self.o, axis = 1)
        print self.maxIndex

    def get_s(self):
        return self.s

    def get_o(self):
        return self.o
       
    def caculate_total_loss(self, x, y):
        loss = 0.
        for i in range(len(y)):
            s, o = self.forward(x[i])
            correct_word_predictions = o[np.arange(len(y[i])), y[i]]
            print correct_word_predictions.shape
            print correct_word_predictions
            loss += -1 * np.sum(np.log(correct_word_predictions))
        return loss
    
    def caculate_loss(self, x, y):
        N = np.sum((len(y_i) for y_i in y))
        print self.caculate_total_loss(x, y)/N

    def bptt(self, x, y):
        length = len(y);
        s, o = self.forward(x)
        dLdU = np.zeros_like(self.U)
        dLdV = np.zeros_like(self.V)
        dLdW = np.zeros_like(self.W)
        delta_o = o
        delta_o[np.arange(len(y)), y] -= 1.
        for t in np.arange(length)[::-1]:
            dLdV += np.outer(delta_o[t]. s[t].T)
            delta_t = self.V.T.dot(delta_o[t]) * (1 - (s[t] ** 2))
    
if __name__ == '__main__':
    getData = getData()
    x, y = getData.encode('./DateSet/reddit-comments-2015-08.csv')
    # print x.shape
    # print y.shape
    # print x[0]
    # print y[0]
    rnn = RNN(8000, 100, 4)
    # rnn.forward(x[:1000])
    # rnn,predict()
    # print rnn.get_o().shape
    # print rnn.get_o()
    rnn.caculate_loss(x[:1000], y[:1000])
