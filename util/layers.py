#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf

sess = tf.InteractiveSession()

# def leaky(layer, alpha=0.1):
#     return tf.maximum(layer, alpha*layer)
#
# a = tf.constant([[1,2,3],[4,1,3]])
#
# v,i = tf.nn.top_k(a,2)
# print(sess.run([v,i,]))
import numpy as np
a = np.array([1,2,3,4])
mask = [True,False,True,False]
b = np.random.uniform(0,10,[4,4])
# for i in range(10):
#     np.random.shuffle(a)
#     print(a)
am = tf.boolean_mask(a, mask)
bm = tf.boolean_mask(b, mask)
print(sess.run(bm))





