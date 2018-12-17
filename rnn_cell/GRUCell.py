#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from pygments.styles import vs
from tensorflow import sigmoid
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _linear
from tensorflow.python.ops import array_ops


def __call__(self, inputs, state, scope=None):
    """Gated recurrent unit (GRU) with nunits cells."""
    with _checked_scope(self, scope or "gru_cell", reuse=self._reuse):
        with vs.variable_scope("gates"):  # Reset gate and update gate.
            # We start with bias of 1.0 to not reset and not update.
            # 一次计算出两个gate的值
            value = sigmoid(_linear([inputs, state], 2 * self._num_units, True, 1.0))
            r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)
        with vs.variable_scope("candidate"):
            c = self._activation(_linear([inputs, r * state], self._num_units, True))

        new_h = u * state + (1 - u) * c
    # GRU里面输出和state都是一个h
    return new_h, new_h
