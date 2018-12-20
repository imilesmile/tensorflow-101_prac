#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import os,sys
import tensorflow as tf
import numpy as np
filePath = './part-r-00199'
for s_example in tf.python_io.tf_record_iterator(filePath):
    example = tf.train.Example()
    example.ParseFromString(s_example)
    print (example)
    break
