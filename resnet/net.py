#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np

# d = np.load('./reskey.npy').item()
class RESNET():
    def __init__(self, resdict, training):
        self.x = tf.placeholder(tf.float32,[None, 416, 416, 3])
        self.feat = self.net(self.x, resdict, training)
        self.var = tf.global_variables()

    def net(self, inputs, resdict, training= False, is_fc= False):
        print('BUILE RES NET ')
        with tf.variable_scope('scale1'):
            w = tf.get_variable('weights',shape=[7,7,3,64])
            x = tf.nn.conv2d(inputs,w,[1,2,2,1],padding='SAME')
        x = tf.layers.batch_normalization(x, training=training,name='scale1')
        x = tf.nn.relu(x)
        x = tf.layers.max_pooling2d(x,3,2,padding='same')

        with tf.variable_scope('scale2'):
            s1_b1_s = blocks(x, 'scale2','block1','shortcut', d=resdict, training=training, activate=None)
            s1_b1_a = blocks(x, 'scale2', 'block1', 'a',d=resdict, training=training)
            s1_b1_b = blocks(s1_b1_a, 'scale2', 'block1', 'b',d=resdict, training=training)
            s1_b1_c = blocks(s1_b1_b, 'scale2', 'block1', 'c', d=resdict,activate=None, training=training)
            s1_b1 = tf.nn.relu(s1_b1_s + s1_b1_c)


            s1_b2_a = blocks(s1_b1, 'scale2', 'block2', 'a', d=resdict, training=training)
            s1_b2_b = blocks(s1_b2_a, 'scale2', 'block2', 'b', d=resdict, training=training)
            s1_b2_c = blocks(s1_b2_b, 'scale2', 'block2', 'c', d=resdict, activate=None, training=training)
            s1_b2_c = tf.nn.relu(s1_b1 + s1_b2_c)

            s1_b3_a = blocks(s1_b2_c, 'scale2', 'block3', 'a', d=resdict, training=training)
            s1_b3_b = blocks(s1_b3_a, 'scale2', 'block3', 'b', d=resdict, training=training)
            s1_b3_c = blocks(s1_b3_b, 'scale2', 'block3', 'c', d=resdict, activate=None, training=training)
            s1_b3_c = tf.nn.relu(s1_b2_c + s1_b3_c)

        with tf.variable_scope('scale3'):
            s = blocks(s1_b3_c, 'scale3', 'block1', 'shortcut', d=resdict, strides=[1,2,2,1],activate=None, training=training)
            x = blocks(s1_b3_c, 'scale3', 'block1', 'a', d=resdict, strides=[1,2,2,1], training=training)
            x = blocks(x, 'scale3', 'block1', 'b', d=resdict, training=training)
            x = blocks(x, 'scale3', 'block1', 'c', d=resdict, activate=None, training=training)
            xx = tf.nn.relu(x + s)

            x = blocks(xx, 'scale3', 'block2', 'a', d=resdict, training=training)
            x = blocks(x, 'scale3', 'block2', 'b', d=resdict, training=training)
            x = blocks(x, 'scale3', 'block2', 'c', d=resdict,activate=None, training=training)
            xx = tf.nn.relu(xx + x)

            x = blocks(xx, 'scale3', 'block3', 'a', d=resdict, training=training)
            x = blocks(x, 'scale3', 'block3', 'b', d=resdict, training=training)
            x = blocks(x, 'scale3', 'block3', 'c', d=resdict,activate=None, training=training)
            xx = tf.nn.relu(xx + x)

            x = blocks(xx, 'scale3', 'block4', 'a', d=resdict, training=training)
            x = blocks(x, 'scale3', 'block4', 'b', d=resdict, training=training)
            x = blocks(x, 'scale3', 'block4', 'c', d=resdict, activate=None, training=training)
            xx3 = tf.nn.relu(xx + x)

        with tf.variable_scope('scale4'):
            s = blocks(xx3, 'scale4', 'block1', 'shortcut', d=resdict, strides=[1,2,2,1],activate=None, training=training)
            x = blocks(xx3, 'scale4', 'block1', 'a', d=resdict, strides=[1,2,2,1], training=training)
            x = blocks(x, 'scale4', 'block1', 'b', d=resdict, training=training)
            x = blocks(x, 'scale4', 'block1', 'c', d=resdict, activate=None, training=training)
            xx = tf.nn.relu(x + s)

            x = blocks(xx, 'scale4', 'block2', 'a', d=resdict, training=training)
            x = blocks(x, 'scale4', 'block2', 'b', d=resdict, training=training)
            x = blocks(x, 'scale4', 'block2', 'c', d=resdict,activate=None, training=training)
            xx = tf.nn.relu(xx + x)

            x = blocks(xx, 'scale4', 'block3', 'a', d=resdict, training=training)
            x = blocks(x, 'scale4', 'block3', 'b', d=resdict, training=training)
            x = blocks(x, 'scale4', 'block3', 'c', d=resdict,activate=None, training=training)
            xx = tf.nn.relu(xx + x)

            x = blocks(xx, 'scale4', 'block4', 'a', d=resdict, training=training)
            x = blocks(x, 'scale4', 'block4', 'b', d=resdict, training=training)
            x = blocks(x, 'scale4', 'block4', 'c', d=resdict,activate=None, training=training)
            xx = tf.nn.relu(xx + x)

            x = blocks(xx, 'scale4', 'block5', 'a', d=resdict, training=training)
            x = blocks(x, 'scale4', 'block5', 'b', d=resdict, training=training)
            x = blocks(x, 'scale4', 'block5', 'c', d=resdict,activate=None, training=training)
            xx = tf.nn.relu(xx + x)

            x = blocks(xx, 'scale4', 'block6', 'a', d=resdict, training=training)
            x = blocks(x, 'scale4', 'block6', 'b', d=resdict, training=training)
            x = blocks(x, 'scale4', 'block6', 'c', d=resdict,activate=None, training=training)
            xx2 = tf.nn.relu(xx + x)

        with tf.variable_scope('scale5'):
            s = blocks(xx2, 'scale5', 'block1', 'shortcut', d=resdict, strides=[1,2,2,1],activate=None, training=training)
            x = blocks(xx2, 'scale5', 'block1', 'a', d=resdict,strides=[1,2,2,1], training=training)
            x = blocks(x, 'scale5', 'block1', 'b', d=resdict, training=training)
            x = blocks(x, 'scale5', 'block1', 'c', d=resdict, training=training)
            xx = tf.nn.relu(x + s)

            x = blocks(xx, 'scale5', 'block2', 'a', d=resdict, training=training)
            x = blocks(x, 'scale5', 'block2', 'b', d=resdict, training=training)
            x = blocks(x, 'scale5', 'block2', 'c', d=resdict, training=training)
            xx = tf.nn.relu(xx + x)

            x = blocks(xx, 'scale5', 'block3', 'a', d=resdict, training=training)
            x = blocks(x, 'scale5', 'block3', 'b', d=resdict, training=training)
            x = blocks(x, 'scale5', 'block3', 'c', d=resdict, training=training)
            xx1 = tf.nn.relu(xx + x)

        if is_fc:
            xx = tf.layers.average_pooling2d(xx,7,1)
            xx = tf.reshape(xx,[-1,2048])
            xx = fc('fc',xx,d=resdict)

        return xx1, xx2, xx3

def get_name(scale,block,branch):
    return scale+'/'+block+'/'+branch+'/'


def blocks(inputs, scale, block, branch, d, strides =[1,1,1,1], bn = True, activate = tf.nn.relu, training = False):
    trainable = training
    name = get_name(scale, block, branch)
    with tf.name_scope(scale):
        with tf.variable_scope(block):
            with tf.variable_scope(branch):
                w = tf.get_variable('weights', shape=d[name + 'weights'], trainable=trainable)
                x = tf.nn.conv2d(inputs,w,strides,padding='SAME')
            if bn:
                x = tf.layers.batch_normalization(x, name=branch, training=training)
            if activate:
                return activate(x)
            if activate == None:
                return x

def fc(scope, inputs, d):
    with tf.variable_scope(scope):
        w = tf.get_variable('weights', shape=d['fc/weights'])
        b = tf.get_variable('biases', shape=d['fc/biases'])
        x = tf.matmul(inputs, w) + b
        return x

def upsample(feat, size= 2):
    shape = feat.get_shape().as_list()
    w,h,c = shape[1:]
    feat = tf.reshape(feat, [-1,h, 1, w, 1, c])
    feat = tf.tile(feat, [1,1,size,1,size,1])
    feat = tf.reshape(feat, [-1, h*size,  w*size, c])
    return feat



if __name__ == '__main__':
    a = tf.placeholder(tf.float32, [3,3,3,3])
    b = upsample(a)
    print(b.shape)



