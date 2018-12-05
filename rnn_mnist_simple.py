#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# Sequence classification with LSTM
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

print ("Packages imported")

mnist = input_data.read_data_sets("data/", one_hot=True)
trainimgs, trainlabels, testimgs, testlabels = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
ntrain, ntest, dim, nclasses = trainimgs.shape[0], testimgs.shape[0], trainimgs.shape[1], trainlabels.shape[1]
print ("MNIST loaded")

# We will treat the MNIST image ∈r28×28∈R28×28 as 2828 sequences of a vector x∈r28
# Our simple RNN consists of
# 1.One input layer which converts a 2828 dimensional input to an 128128 dimensional hidden layer,
# 2.One intermediate recurrent neural network (LSTM)
# 3.One output layer which converts an 128128 dimensional output of the LSTM to 1010 dimensional output indicating a class label.
# Contruct a Recurrent Neural Network
diminput = 28
dimhidden = 128
dimoutput = nclasses
nsteps = 28
weights = {
    'hidden': tf.Variable(tf.random_normal([diminput, dimhidden])),
    'out': tf.Variable(tf.random_normal([dimhidden, dimoutput]))
}
biases = {
    'hidden': tf.Variable(tf.random_normal([dimhidden])),
    'out': tf.Variable(tf.random_normal([dimoutput]))
}


def _RNN(_X, _istate, _W, _b, _nsteps, _name):
    # 1. Permute input from [batchsize, nsteps, diminput] => [nsteps, batchsize, diminput]
    _X = tf.transpose(_X, [1, 0, 2])
    # 2. Reshape input to [nsteps*batchsize, diminput]
    _X = tf.reshape(_X, [-1, diminput])
    # 3. Input layer => Hidden layer
    _H = tf.matmul(_X, _W['hidden'] + _b['hidden'])
    # 4. Splite data to 'nsteps' chunks. An i-th chunck indicates i-th batch data
    _Hsplit = tf.split(0, _nsteps, _H)
    # 5. Get LSTM's final output (_LSTM_O) and state (_LSTM_S)
    #    Both _LSTM_O and _LSTM_S consist of 'batchsize' elements
    #    Only _LSTM_O will be used to predict the output.
    with tf.variable_scope(_name):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(dimhidden, forget_bias=1.0)
        _LSTM_O, _LSTM_S = tf.nn.rnn(lstm_cell, _Hsplit, initial_state=_istate)
    # 6. Output
    _O = tf.matmul(_LSTM_O[-1], _W['out']) + _b['out']
    # Return!
    return {'X': _X, 'H': _H, 'Hsplit': _Hsplit, 'LSTM_O': _LSTM_O, 'LSTM_S': _LSTM_S, 'O': _O}


print ("Network ready")

# define function
learning_rate = 0.001
x = tf.placeholder("float", [None, nsteps, diminput])
istate = tf.placeholder("float", [None, 2 * dimhidden])  # state & cell => 2x n_hidden
y = tf.placeholder("float", [None, dimoutput])
myrnn = _RNN(x, istate, weights, biases, nsteps, 'basic')
pred = myrnn['O']

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optm = tf.train.AdamOptimizer(learning_rate).minimize(cost)  # Adam Optimizer
accr = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), tf.float32))
init = tf.initialize_all_variables()
print ("Network Ready!")

# run
training_epochs = 5
batch_size = 128
display_step = 1
sess = tf.Session()
sess.run(init)
summary_writer = tf.train.SummaryWriter('/tmp/tensorflow_logs', graph=sess.graph)
print ("Start optimization")
for epoch in range(training_epochs):
    avg_cost = 0.
    total_batch = int(mnist.train.num_examples / batch_size)
    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, nsteps, diminput))
        # Fit training using batch data
        feeds = {x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2 * dimhidden))}
        sess.run(optm, feed_dict=feeds)
        # Compute average loss
        avg_cost += sess.run(cost, feed_dict=feeds) / total_batch
    # Display logs per epoch step
    if epoch % display_step == 0:
        print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
        feeds = {x: batch_xs, y: batch_ys, istate: np.zeros((batch_size, 2 * dimhidden))}
        train_acc = sess.run(accr, feed_dict=feeds)
        print (" Training accuracy: %.3f" % (train_acc))
        testimgs = testimgs.reshape((ntest, nsteps, diminput))
        feeds = {x: testimgs, y: testlabels, istate: np.zeros((ntest, 2 * dimhidden))}
        test_acc = sess.run(accr, feed_dict=feeds)
        print (" Test accuracy: %.3f" % (test_acc))
print ("Optimization Finished.")
# Start optimization
# Epoch: 000/005 cost: 0.479400075
#  Training accuracy: 0.992
#  Test accuracy: 0.922
# Epoch: 001/005 cost: 0.136942688
#  Training accuracy: 0.969
#  Test accuracy: 0.959
# Epoch: 002/005 cost: 0.081425477
#  Training accuracy: 0.984
#  Test accuracy: 0.951
# Epoch: 003/005 cost: 0.061170839
#  Training accuracy: 0.969
#  Test accuracy: 0.973
# Epoch: 004/005 cost: 0.047727333
#  Training accuracy: 0.992
#  Test accuracy: 0.973
# Optimization Finished.


# How may sequences will we use?
nsteps2 = 25

# Test with truncated inputs
testimgs = testimgs.reshape((ntest, nsteps, diminput))
testimgs_trucated = np.zeros(testimgs.shape)
testimgs_trucated[:, 28 - nsteps2:] = testimgs[:, :nsteps2, :]
feeds = {x: testimgs_trucated, y: testlabels, istate: np.zeros((ntest, 2 * dimhidden))}
test_acc = sess.run(accr, feed_dict=feeds)
print (" If we use %d seqs, test accuracy becomes %.3f" % (nsteps2, test_acc))
#  If we use 25 seqs, test accuracy becomes 0.766

# What's going on inside the RNN?
# inputs to the rnn
batch_size = 5
xtest, _ = mnist.test.next_batch(batch_size)
print ("Shape of 'xtest' is %s" % (xtest.shape,))
# Shape of 'xtest' is (5, 784)

# Reshaped inputs
# Reshape (this will go into the network)
xtest1 = xtest.reshape((batch_size, nsteps, diminput))
print ("Shape of 'xtest1' is %s" % (xtest1.shape,))
# Shape of 'xtest1' is (5, 28, 28)

# Feeds: inputs and initial states
feeds = {x: xtest1, istate: np.zeros((batch_size, 2 * dimhidden))}

# Each indivisual input to the LSTM
rnnout_X = sess.run(myrnn['X'], feed_dict=feeds)
print ("Shape of 'rnnout_X' is %s" % (rnnout_X.shape,))
# Shape of 'rnnout_X' is (140, 28)

# Each indivisual intermediate state
rnnout_H = sess.run(myrnn['H'], feed_dict=feeds)
print ("Shape of 'rnnout_H' is %s" % (rnnout_H.shape,))
# Shape of 'rnnout_H' is (140, 128)

# Actual input to the LSTM (List)
rnnout_Hsplit = sess.run(myrnn['Hsplit'], feed_dict=feeds)
print ("Type of 'rnnout_Hsplit' is %s" % (type(rnnout_Hsplit)))
print ("Length'rnnout_Hsplit' is %s and the shape of each item is %s" % (len(rnnout_Hsplit), rnnout_Hsplit[0].shape))
# Type of 'rnnout_Hsplit' is <type 'list'>
# Length of 'rnnout_Hsplit' is 28 and the shape of each item is (5, 128)

# Output from the LSTM (List)
rnnout_LSTM_O = sess.run(myrnn['LSTM_O'], feed_dict=feeds)
print ("Type of 'rnnout_LSTM_O' is %s" % (type(rnnout_LSTM_O)))
print ("Length 'rnnout_LSTM_O' is %s and the shape of each item is %s" % (len(rnnout_LSTM_O), rnnout_LSTM_O[0].shape))

# Final prediction
rnnout_O = sess.run(myrnn['O'], feed_dict=feeds)
print ("Shape of 'rnnout_O' is %s" % (rnnout_O.shape,))
# Shape of 'rnnout_O' is (5, 10)
