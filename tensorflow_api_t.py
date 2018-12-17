#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow import int32



sess = tf.Session()
# ==============================================================================================================================================================
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

# embedding lookup
# tf.nn.embedding_lookup的作用就是找到要寻找的embedding data中的对应的行下的vector
# tf.nn.embedding_lookup(params, ids, partition_strategy='mod', name=None, validate_indices=True, max_norm=None)
data = np.array([[[2], [1]], [[3], [4]], [[6], [7]]])
print (data)
# tf.convert_to_tensor转化我们现有的array
data = tf.convert_to_tensor(data)
print (data)
lk = [[0, 1], [1, 0], [0, 0]]
lookup_data = tf.nn.embedding_lookup(data, lk)
init = tf.global_variables_initializer()
# lk中的值，在要寻找的embedding数据中下找对应的index下的vector进行拼接。
# 永远是look(lk)部分的维度+embedding(data)部分的除了第一维后的维度拼接。
# 很明显，我们也可以得到，lk里面值是必须要小于等于embedding(data)的最大维度减一的
print (np.array(lk).shape)
print (lookup_data)
print (sess.run(lookup_data))

# trainable_variables 和 all_variables
# tf.trainable_variables返回的是需要训练的变量列表
# tf.all_variables返回的是所有变量的列表
# 在创造变量(tf.Variable, tf.get_variable 等操作)时，都会有一个trainable的选项，表示该变量是否可训练。
# 这个函数会返回图中所有trainable=True的变量。
# tf.get_variable(…), tf.Variable(…)的默认选项是True, 而 tf.constant(…)只能是False


v = tf.Variable(tf.constant(0.0, shape=[1], dtype=tf.float32), name='v')
v1 = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='v1')

global_step = tf.Variable(tf.constant(5, shape=[1], dtype=tf.float32), name='global_step', trainable=False)
# ExponentialMovingAverage来实现滑动平均模型，他使用指数衰减来计算变量的移动平均值
ema = tf.train.ExponentialMovingAverage(0.99, global_step)

for ele1 in tf.trainable_variables():
    print (ele1.name)
print("=" * 18)
for ele2 in tf.all_variables():
    print (ele2.name)
# ==============================================================================================================================================================
# 矩阵操作
# ones 和 zeros
# tf.InteractiveSession():它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。这对于工作在交互式环境中的人们来说非常便利，比如使用IPython。
# tf.Session():需要在启动session之前构建整个计算图，然后启动该计算图。
ia_sess = tf.InteractiveSession()
x = tf.ones([2, 3], int32)
print (ia_sess.run(x))

# ones_like 和 zeros_like
# 新建一个与给定的tensor类型大小一致的tensor，其所有元素为1和0
tensor = [[1, 2, 3], [4, 5, 6]]
x = tf.ones_like(tensor)
print (ia_sess.run(x))

# fill
# 创建一个形状大小为shape的tensor，其初始值为value
print (ia_sess.run(tf.fill([2, 3], 2)))

# tf.random_normal | tf.truncated_normal | tf.random_uniform
# 这几个都是用于生成随机数tensor的。尺寸是shape
# random_normal: 正太分布随机数，均值mean,标准差stddev
# truncated_normal:截断正态分布随机数，均值mean,标准差stddev,不过只保留[mean-2*stddev,mean+2*stddev]范围内的随机数
# random_uniform:均匀分布随机数，范围为[minval,maxval]
x = tf.random_normal(shape=[1, 5], mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
print (sess.run(x))
# initializer: 初始化工具，有tf.zero_initializer, tf.ones_initializer, tf.constant_initializer,
# tf.random_uniform_initializer, tf.random_normal_initializer, tf.truncated_normal_initializer等

# 1.2 矩阵变换
# shape
# tf.shape函数本身也是返回一个张量。而在tf中，张量是需要用sess.run(Tensor)来得到具体的值的
labels = [1, 2, 3]
shape = tf.shape(labels)
print (shape)
print (ia_sess.run(shape))

# expand_dims
x = tf.expand_dims(labels, 0)
print (sess.run(x))
x = tf.expand_dims(labels, 1)
print(sess.run(x))

# stack
# 将一个R维张量列表沿着axis轴组合成一个R+1维的张量。tf.pack改为了tf.stack
x = [1, 4]
y = [2, 5]
z = [3, 6]
s = tf.stack([x, y, z], axis=1)
print (ia_sess.run(s))

# concat
# 将张量沿着指定维数拼接起来。个人感觉跟前面的pack用法类似
# t1 = [[1, 2, 3], [4, 5, 6]]
# t2 = [[7, 8, 9], [10, 11, 12]]
# m = tf.concat(0, [t1, t2])
# n = tf.concat(1, [t1, t2])

# random_shuffle
#  沿着value的第一维进行随机重新排列
a = [[1, 2], [3, 4], [5, 6]]
x = tf.random_shuffle(a)
print(sess.run(x))

# argmax |argmin
# 找到给定的张量tensor中在指定轴axis上的最大值/最小值的位置。
a = tf.get_variable(name='a', shape=[3, 4], dtype=tf.float32,
                    initializer=tf.random_uniform_initializer(minval=-1, maxval=1))
b = tf.argmax(input=a, dimension=0)
c = tf.argmax(input=a, dimension=1)
sess.run(tf.initialize_all_variables())
print(sess.run(a))
# [[-0.02054167  0.35913634 -0.00463343  0.5969434 ]
#  [-0.34672737  0.19666076 -0.34608603  0.71880126]
#  [ 0.2980399  -0.27219176 -0.49712944 -0.91940784]]
print(sess.run(b))
# [2 0 0 1]
print(sess.run(c))
# [3 3 0]

# equal
# 判断两个tensor是否每个元素都相等。返回一个格式为bool的tensor

# cast
# 将x的数据格式转化成dtype.
a = tf.Variable([1, 0, 0, 1, 1])
b = tf.cast(a, dtype=tf.bool)
sess.run(tf.initialize_all_variables())
print(sess.run(b))

# tf.matmul
# 用来做矩阵乘法。若a为l*m的矩阵，b为m*n的矩阵，那么通过tf.matmul(a,b) 结果就会得到一个l*n的矩阵

# reshape
# 就是将tensor按照新的shape重新排列。一般来说，shape有三种用法：
# 如果 shape=[-1], 表示要将tensor展开成一个list
# 如果 shape=[a,b,c,…] 其中每个a,b,c,..均>0，那么就是常规用法
# 如果 shape=[a,-1,c,…] 此时b=-1，a,c,..依然>0。这表示tf会根据tensor的原尺寸，自动计算b的值。
# t = tf.Variable([1, 2, 3, 4, 5, 6, 7, 8, 9])
# r1 = tf.reshape(t, [3, 3])
# print (r1)
# # tensor 't' is [[[1, 1], [2, 2]],
#                [[3, 3], [4, 4]]]
# tensor 't' has shape [2, 2, 2]
# r1 = tf.reshape(t, [2, 4])
# == > [[1, 1, 2, 2],
# [3, 3, 4, 4]]

# tensor 't' is [[[1, 1, 1],
#                 [2, 2, 2]],
#                [[3, 3, 3],
#                 [4, 4, 4]],
#                [[5, 5, 5],
#                 [6, 6, 6]]]
# tensor 't' has shape [3, 2, 3]
# pass '[-1]' to flatten 't'
# r1 = tf.reshape(t, [-1])
# # == > [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 6, 6, 6]
#
# # -1 can also be used to infer the shape
# # -1 is inferred to be 9:
# r1 = tf.reshape(t, [2, -1])
# # == > [[1, 1, 1, 2, 2, 2, 3, 3, 3],
# # [4, 4, 4, 5, 5, 5, 6, 6, 6]]
#
# # -1 is inferred to be 2:
# r1 = tf.reshape(t, [-1, 9])
# # == > [[1, 1, 1, 2, 2, 2, 3, 3, 3],
# # [4, 4, 4, 5, 5, 5, 6, 6, 6]]
#
# # -1 is inferred to be 3:
# r1 = tf.reshape(t, [2, -1, 3])
# # == > [[[1, 1, 1],
#  [2, 2, 2],
#  [3, 3, 3]],
# [[4, 4, 4],
#  [5, 5, 5],
#  [6, 6, 6]]]

# 2. 神经网络相关
# tf.clip_by_global_norm
# 修正梯度值，用于控制梯度爆炸的问题。梯度爆炸和梯度弥散的原因一样，都是因为链式法则求导的关系，导致梯度的指数级衰减。
# 为了避免梯度爆炸，需要对梯度进行修剪。
# def clip_by_global_norm(t_list, clip_norm, use_norm=None, name=None):
# t_list为待修剪的张量, clip_norm 表示修剪比例(clipping ratio).

# tf.nn.dropout
# 按概率来将x中的一些元素值置零，并将其他的值放大。用于进行dropout操作，一定程度上可以防止过拟合

# 3.普通操作
# tf.linspace | tf.range
# 这两个放到一起说，是因为他们都用于产生等差数列
# tf.linspace在[start,stop]范围内产生num个数的等差数列。不过注意，start和stop要用浮点数表示，不然会报错
# tf.range在[start,limit)范围内以步进值delta产生等差数列。注意是不包括limit在内的。
x = tf.linspace(start=1.0, stop=5.0, num=5, name=None)  # 注意1.0和5.0
y = tf.range(start=1, limit=5, delta=1)
print(sess.run(x))
print(sess.run(y))
# ===>[ 1.  2.  3.  4.  5.]
# ===>[1 2 3 4]

# tf.assign
# tf.assign是用来更新模型中变量的值的。ref是待赋值的变量，value是要更新的值。即效果等同于 ref = value
a = tf.Variable(0.0)
b = tf.placeholder(dtype=tf.float32, shape=[])
op = tf.assign(a, b)
sess.run(tf.initialize_all_variables())
print(sess.run(a))
# 0.0
sess.run(op, feed_dict={b: 5.})
print(sess.run(a))

# 4.规范化
# tf.variable_scope
# tf.get_variable_scope

# x.get_shape().as_list()
# 第一点：tensor.get_shape()返回的是元组，不能放到sess.run()里面，这个里面只能放operation和tensor；
# 第二点：tf.shape(）返回的是一个tensor。要想知道是多少，必须通过sess.run()
a_array = np.array([[1, 2, 3], [4, 5, 6]])
b_list = [[1, 2, 3], [4, 5, 6]]
c_tensor = tf.constant([[1, 2, 3], [4, 5, 6]])

print(c_tensor.get_shape())
print(c_tensor.get_shape().as_list())

with tf.Session() as sess:
    print(sess.run(tf.shape(a_array)))
    print(sess.run(tf.shape(b_list)))
    print(sess.run(tf.shape(c_tensor)))

# nest.flatten()
# 将嵌套结构压平，返回Python的list。
from tensorflow.python.util import nest
input = [['a', 'b', 'c'],
        ['d', 'e', 'f'],
        ['1', '2', '3']]

result = nest.flatten(input)