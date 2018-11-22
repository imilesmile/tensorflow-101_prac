#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf

print ("PACKAGES LOADED")

sess = tf.Session()
print ("open session")


# tf constant
def print_tf(x):
    print("TYPE IS\n %s" % (type(x)))
    print("VALUE IS\n %s" % (x))


hello = tf.constant("hello it's me")
print_tf(hello)

# to make thinks happen
hello_out = sess.run(hello)
print_tf(hello_out)

# other types of constant
a = tf.constant(1.5)
b = tf.constant(2.5)
print_tf(a)
print_tf(b)
a_out = sess.run(a)
b_out = sess.run(b)
print_tf(a_out)
print_tf(b_out)

# operators
a_plus_b = tf.add(a, b)
print_tf(a_plus_b)
a_plus_b_out = sess.run(a_plus_b)
print_tf(a_plus_b_out)

a_mul_b = tf.multiply(a, b)
a_mul_b_out = sess.run(a_mul_b)
print_tf(a_mul_b_out)

# variables
weight = tf.Variable(tf.random_normal([5, 2], stddev=0.1))
print_tf(weight)

# #error because not initialize
# weight_out = sess.run(weight)
# print_tf(weight_out)

init = tf.global_variables_initializer()
sess.run(init)
print("initial all variables")

weight_out = sess.run(weight)
print_tf(weight_out)

x = tf.placeholder(tf.float32, [None, 5])
print_tf(x)

# operation with variables
oper = tf.matmul(x, weight)
print_tf(oper)

data = np.random.rand(1, 5)
oper_out = sess.run(oper, feed_dict={x: data})
print_tf(oper_out)

data = np.random.rand(2, 5)
oper_out = sess.run(oper, feed_dict={x: data})
print_tf(oper_out)
print(oper_out)
