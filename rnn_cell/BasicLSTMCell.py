#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# 1.BasicRNNCell，也就是经典的RNN，其调用的时候输出和状态的计算公式是：
# output = new_state = act(W input + U state + B)，其内部调用了_linear()函数。
# 2._linear()函数，接受输入，并将输入与参数矩阵W相乘，加上偏置b，并返回。
from tensorflow import sigmoid
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.rnn_cell_impl import LSTMStateTuple


def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell(LSTM)"""
    with _checked_scope(self, scope or "basic_lstm_cell", reuse=self._reuse):
        # parameters of gates are concated into one multiply for efficiency
        if self._state_is_tuple:
            # 一般都走这个分支，取出c_t和h_t
            c, h = state
        else:
            c, h = array_ops.split(value=concat, num_or_size_splits=4, axis=1)
            # 参考了《Recurrent Neural Network Regularization》，一次计算四个gate
    concat = _linear([inputs, h], 4 * self._num_units, True)

    # i = input_gate, j = new_input, f = forget_gate, o = output_gate
    i, j, f, o = array_ops.split(value=concat, num_or_size_splits=4, axis=1)

    new_c = (c* sigmoid(f+self._forget_bias)+ sigmoid(i)* self._activation(j))

    new_h = self._activation(new_c)*sigmoid(o)

    if self._state_is_tuple:
      new_state = LSTMStateTuple(new_c, new_h)
    else:
      new_state = array_ops.concat([new_c, new_h], 1)
    # 注意这里返回的输出是h_t,而state是(c,h)
    return new_h, new_state

