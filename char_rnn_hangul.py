#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Import Packages
import numpy as np
import tensorflow as tf
import collections
import string
import argparse
import time
import os
from six.moves import cPickle

# Load dataset using TextLoader
data_dir = "data/nine_dreams"
batch_size = 50
seq_length = 50
data_loader = ""
vocab_size = data_loader.vocab_size
vocab = data_loader.vocab
chars = data_loader.chars
print ("type of 'data_loader' is %s, length is %d"
       % (type(data_loader.vocab), len(data_loader.vocab)))
print ("\n")
print ("data_loader.vocab looks like \n%s " %
       (data_loader.vocab))
print ("\n")
print ("type of 'data_loader.chars' is %s, length is %d"
       % (type(data_loader.chars), len(data_loader.chars)))
print ("\n")
print ("data_loader.chars looks like \n%s " % (data_loader.chars,))

# define network
rnn_size = 512
num_layers = 3
grad_clip = 5.

_batch_size = 1
_seq_length = 1

with tf.device("/cpu:0"):
    # select rnn cell
    unitcell = tf.nn.rnn_cell.BasicLSTMCell(rnn_size)
    cell = tf.nn.rnn_cell.MultiRNNCell([unitcell] * num_layers)
    # set paths to the graph
    input_data = tf.placeholder(tf.int32, [_batch_size, _seq_length])
    targets = tf.placeholder(tf.int32, [_batch_size, _seq_length])
    initial_state = cell.zero_state(_batch_size, tf.float32)

    # set network
    with tf.variable_scope("rnnlm"):
        softmax_w = tf.get_variable("softmax_w", [rnn_size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, rnn_size])
            inputs = tf.split(1, _seq_length, tf.nn.embedding_lookup(embedding, input_data))
            inputs = [tf.squeeze(input_, [1]) for input_ in inputs]
    #loop
