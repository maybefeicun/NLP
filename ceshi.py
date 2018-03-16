# -*- coding: utf-8 -*-
# @Time    : 2018/3/14 15:09
# @Author  : chen
# @File    : ceshi.py
# @Software: PyCharm

import tensorflow as tf
import numpy as np

sess = tf.Session()

output = round(10 * 0.8)
print(output)

A = tf.ones(shape=4)
print(sess.run(A))
B = tf.diag(A)
print(sess.run(B))