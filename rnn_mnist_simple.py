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

