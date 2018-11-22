#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import numpy as np

a = np.array([1, 5, 3])
b = np.array([2, 3, 4])
print (a.shape)

c = np.stack((a), axis=0)
print (c.shape)

d = np.stack((a, b), axis=0)
print (d.dtype)
