#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import tensorflow as tf

# ckpt......
## save
inputs = tf.placeholder(tf.float32, shape=[None, 128], name='inputs')  #<-- 入口
prediction = tf.nn.softmax('logits', name='prediction')  #<-- 出口（仅作为例子，下同）
saver = tf.train.Saver()

with tf.Session() as sess:
    # <-- 训练过程
    saver.save(sess, './xxx/xxx.ckpt')  #<-- 模型保存

## restore
with tf.Session() as sess:
    saver = tf.train.import_meta_graph('./xxx/xxx.ckpt.meta')
    saver.restore(sess, './xxx/xxx.ckpt')

    inputs = tf.get_default_graph().get_tensor_by_name('inputs:0')
    prediction = tf.get_default_graph().get_tensor_by_name('prediction:0')

    pred = sess.run(prediction, feed_dict={inputs : xxx})

# pb.....
#  .pb 格式则不能继续训练，因为这种格式保存的模型参数都已经转化为了常量（而不再是变量）
## save
from tensorflow.python.framework import graph_util

inputs = tf.placeholder(tf.float32, shape=[None, 128], name = 'inputs')

prediction = tf.nn.softmax('logits', name='prediction')

with tf.Session() as sess:
    #... 训练
    graph_def = tf.get_default_graph().as_graph_def()
    out_graph_def = graph_util.convert_variables_to_constants(
        sess,
        graph_def,
        ['prediction'] #<-- 参数：output_node_names，输出节点名
    )
    with tf.gfile.GFile('./xxx/xxx.pb', 'wb') as fid:
        serialized_graph = out_graph_def.SerializeToString()
        fid.write(serialized_graph)

## restore
import os
def load_model(model_pb_path):
    if not os.path.exists(model_pb_path.pb):
        raise ValueError("'path_to_model.pb' is not exist.")

    model_graph = tf.Graph()
    with model_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(model_pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
        return model_graph


# ckpt 格式转 pb 格式
from tensorflow.python.framework import graph_util

with tf.Session() as sess:
    #load .ckpt file
    saver = tf.train.import_meta_graph('./xxx/xxx.ckpt.meta')
    saver.restore(sess, './xxx/xxx.ckpt')

    #save as .pb file
    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def,['prediction'])
    with tf.gfile.GFile('./xxx/xxx.pb', 'wb') as fid:
        serialized_graph = output_graph_def.SerializeToString()
        fid.write(serialized_graph)


