#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf

sess = tf.Session()

# one-hot
one_hot = tf.one_hot(indices=[0, 2, -1, 1],
                     depth=3,
                     on_value=1.0,
                     off_value=0.0,
                     axis=-1)
print (sess.run(one_hot))

# sequence mask
# 这个操作和one hot也很像，但是指定的不是index而是从前到后有多少个True，返回的是True和False
sq_mask = tf.sequence_mask([1, 3, 2], 5)
print (sess.run(sq_mask))

# boolean_mask
# 这个操作可以用于留下指定的元素，类似于numpy的操作
import numpy as np

tensor = tf.range(4)
print (sess.run(tensor))
mask = np.array([True, False, True, False])
bool_mask = tf.boolean_mask(tensor, mask)
print (sess.run(bool_mask))

# 也可以先用数字传进来，再转换成bool，这样就可以利用one_hot了。
num_mask = np.array([1, 0, 1, 0])
num_mask = tf.cast(num_mask, tf.bool)
bool_num_mask = tf.boolean_mask(tensor, num_mask)
print (sess.run(bool_num_mask))

# mask和被处理的tensor必须shape相同，执行下面的代码会报错：
# tensor = tf.reshape(tf.range(8), [2, 4])
# mask = np.array([True, False, True, False])
# bool_mask = tf.boolean_mask(tensor, mask)
# print (sess.run(bool_mask))
# # ValueError: Shapes (2,) and (4,) are incompatible

# split
# 分割数据
m1 = tf.reshape(tf.range(24), [2, 3, 4])
print (m1)
# print (sess.run(m1))
# tf.split(value, num_or_size_splits, axis=0, num=None, name='split')
split0, split1, split2 = tf.split(m1, 3, 1)
print (split0.get_shape())
# print (sess.run(split0))

# concat
m2 = tf.reshape(tf.range(24), [2, 3, 4])
# print (sess.run(m2))

concat1 = tf.concat([m1, m2], 1)
print (sess.run(concat1))
print (concat1)

# squeeze
# 压缩长度为1 的维度
arr = tf.truncated_normal([3, 2, 1, 6, 1], stddev=0.1)
print (arr.get_shape())
sequeeze_arr = tf.squeeze(arr).get_shape()
print (sequeeze_arr)

# expand_dims
# 和squeeze相反，可以扩展指定的维度。
expand_arr = tf.expand_dims(arr, 0).get_shape()
print (expand_arr)

# gather
# 一个tensor当源数据，一个tensor当下标，取出对应的数据。
indices = tf.placeholder(tf.int32, [5])
arr = tf.range(10, 20)
print (sess.run(arr))
g = tf.gather(arr, indices)
print (sess.run(g, feed_dict={indices: [4, 5, 7, 1, 2]}))

# tile
# 给定一个tensor，堆成更大的tensor。
# tf.tile(input, multiples, name=None)
t_simple = tf.range(10)
t_complex = tf.tile(t_simple, [2])
print (sess.run(t_complex))

t_simple_2 = tf.reshape(tf.range(10), [2, 5])  # multiples的维度和输入的维度需要保持一致
print (sess.run(t_simple_2))
t_complex_2 = tf.tile(t_simple_2, [2, 3])
print (sess.run(t_complex_2))