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

vec = [1] * 5
vec2 = [0] * 5
vec3 = vec + vec2

vec = tf.constant([[1, 2], [3, 4], [5, 6]])
output = tf.reduce_sum(vec, axis=1)
output2 = tf.reduce_sum(vec, axis=1, keep_dims=True)
vec2 = tf.constant([2, 2, 2])
vec2 = tf.expand_dims(vec2, 1)
output3 = vec / vec2
print(sess.run(output3))
pass