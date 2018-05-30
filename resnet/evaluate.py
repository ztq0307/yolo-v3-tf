#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
from resnet import net
import cv2

VGG_MEAN = [103.939, 116.779, 123.68]

class SOLVER():
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
        self.dict = np.load('./reskey.npy').item()
        self.idx2_ = np.load('./idx2desc.npy').item()
        self.logits = net.net(inputs=self.inputs, resdict=self.dict)

    def imgprocess(self, path):
        img = cv2.imread(path)
        imgcopy = np.copy(img)
        img = cv2.resize(img, (416, 416))
        img = np.reshape(img, [1,416,416,3])
        img = img*1.
        img -= VGG_MEAN
        return imgcopy, img

    def evaluate(self, path):
        save = tf.train.Saver()
        save.restore(self.sess, 'D:\CODE5.24\ResYOLO\ResWeight\ResNet-L50.ckpt')
        copy, img = self.imgprocess(path)
        result = tf.nn.softmax(self.logits)[0,:]
        value, indice = tf.nn.top_k(result, 5)
        # print(value.shape)
        v,idx = self.sess.run([value, indice], feed_dict={self.inputs:img})
        for i in range(5):
            print('The Prob is %f ,object is : '%(v[i]),self.idx2_[idx[i]])
        cv2.imshow('w', copy)
        cv2.waitKey()


if __name__ == '__main__':
    solver = SOLVER()
    solver.evaluate('D:\CODE5.24\ResYOLO\\r3.jpg')






