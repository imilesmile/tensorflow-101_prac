#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from pygments.styles import vs
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow.python.data.util import nest
from tensorflow.python.ops import array_ops, nn_ops, math_ops


def __call__(self, inputs, state, scope=None):
    """Long short-term memory cell with attention (LSTMA)."""
    if self._state_is_tuple:
        # 这里把state分为三个部分，LSTM的state，attns（代表attention向量）和attn的state
        state, attns, attn_states = state
    else:
        # 如果不是元组，就按照长度切分
        states = state
        state = array_ops.slice(states, [0, 0], [-1, self._cell.state_size])
        attns = array_ops.slice(
            states, [0, self._cell.state_size], [-1, self._attn_size])
        attn_states = array_ops.slice(
            states, [0, self._cell.state_size + self._attn_size],
            [-1, self._attn_size * self._attn_length])

    # attention状态是[None x Attention向量长度 x Attention窗口长度]
    attn_states = array_ops.reshape(attn_states, [-1, self._attn_length, self._atten_size])
    input_size = self._input_size

    if input_size is None:
        input_size = inputs.get_shape().as_list()[1]
    # 让input 和 attns 进行一个什么运算呢?
    inputs = _linear([inputs, attns], input_size, True)
    lstm_output, new_state = self._cell(inputs, state)
    if self._state_is_tuple:
        new_state_cat = array_ops.concat(nest.flatten(new_state), 1)
    else:
        new_state_cat = new_state

        # 利用attention机制计算出下一时刻需要的上下文向量c_t和attention状态（隐藏状态）h_j
        new_attns, new_attn_states = self._attention(new_state_cat, attn_states)
        with vs.variable_scope("attn_output_projection"):
            # 利用c_t和x_t(y_{t-1})计算出t时刻输出s_t
            output = _linear([lstm_output, new_state], self._atten_size,True)
        # 把当前时刻输出s_t增加到下一时刻attention状态去
        new_attn_states = array_ops.concat([new_attn_states, array_ops.expand_dims(output,1)],1)
        new_attn_states = array_ops.reshape(new_attn_states, [-1, self._attn_length * self._attn_size])
        new_state = (new_state, new_attns, new_attn_states)
        if not self._state_is_tuple:
            new_state = array_ops.concat(list(new_state), 1)
        return output, new_state



def _attention(self, query, atten_states):
    conv2d = nn_ops.conv2d
    reduce_sum = math_ops.reduce_sum
    softmax = nn_ops.softmax
    tanh = math_ops.tanh

    with vs.variable_scope("attention"):
        k = vs.get_variable("attn_W", [1, 1, self._atten_size, self._atten_vec_size]
        v = vs.get_variable("attn_v", [self._attn_vec_size])
        # 相当于所有的h_j
        hidden = array_ops.reshape(atten_states, [-1, self._atten_lenght, 1, self._atten_size])
        # 计算Uh_j,shape:[[None, attn_len, 1, attn_vec_size]]
        hidden_features = nn_ops.conv2d(hidden, k, [1, 1, 1, 1], "SAME")
        y = _linear(query, self._atten_vec_size, True)
        # 计算WS_i
        y = array_ops.reshape(y, [-1, 1, 1, self._attn_vec_size])
        # attention相似度计算公式，s\in R^{-1, attn_len}，对应所有的e_{ij}
        s = nn_ops.reduce_sum(v * tanh(hidden_features + y), [2, 3])
        # a \in R^{-1, attn_len}，对应论文中的\alpha
        a = nn_ops.softmax(s)
        # 计算上下文向量c_i=\sum \alpha_{ij} * h_j
        d = nn_ops.reduce_sum(array_ops.reshape(a, [-1, self._attn_length, 1, 1]) * hidden, [1, 2])
        new_attns = array_ops.reshape(d, [-1, self._attn_size])
        # 扔掉最早的一个attention-states
        new_attn_states = array_ops.slice(attn_states, [0, 1, 0], [-1, -1, -1])
        return new_attns, new_attn_states


