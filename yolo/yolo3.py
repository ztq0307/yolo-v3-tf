#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
from util.cfg import *
from util.boxes import *
from resnet.net import upsample

class YOLO2(object):
    def __init__(self):
        self.CLASS_SCALE = class_scale
        self.OBJECT_SCALE = object_scale
        self.NOOBJECT_SCALE = noobject_scale
        self.COORD_SCALE = coord_scale

    def head(self, feat1, feat2, feat3, num_class, num_anchor, training=True):
        with tf.variable_scope('yolohead'):
            with tf.variable_scope('scale1'):
                layer = tf.layers.conv2d(feat1, 256, 1, padding='same', use_bias=False)
                layer = tf.layers.batch_normalization(layer, training=training)
                layer = self.leaky_relu(layer)
                layer = tf.layers.conv2d(layer, 512, 3, padding='same', use_bias=False)
                layer = tf.layers.batch_normalization(layer, training=training)
                layer = self.leaky_relu(layer)
                layer = tf.layers.conv2d(layer, 256, 1, padding='same', use_bias=False)
                layer = tf.layers.batch_normalization(layer, training=training)
                layer = self.leaky_relu(layer)
                layer = tf.layers.conv2d(layer, 256, 3, padding='same')
                c1 = self.leaky_relu(layer)
                pred1 = tf.layers.conv2d(c1, num_anchor * (num_class + 5), 1, 1, activation=None)

            with tf.variable_scope('scale2'):
                layer = tf.layers.conv2d(feat2, 256, 1, padding='same', use_bias=False)
                layer = tf.layers.batch_normalization(layer, training=training)
                layer = self.leaky_relu(layer)
                layer = tf.layers.conv2d(layer, 512, 3, padding='same', use_bias=False)
                layer = tf.layers.batch_normalization(layer, training=training)
                layer = self.leaky_relu(layer)
                c1_up = upsample(c1, 2)
                layer = tf.concat([layer, c1_up], axis=-1)
                layer = self.leaky_relu(layer)
                layer = tf.layers.conv2d(layer, 256, 1, padding='same', use_bias=False)
                layer = tf.layers.batch_normalization(layer, training=training)
                layer = self.leaky_relu(layer)
                layer = tf.layers.conv2d(layer, 256, 3, padding='same')
                c2 = self.leaky_relu(layer)
                pred2 = tf.layers.conv2d(c2, num_anchor * (num_class + 5), 1, 1, activation=None)

            with tf.variable_scope('scale3'):
                layer = tf.layers.conv2d(feat3, 128, 1, padding='same', use_bias=False)
                layer = tf.layers.batch_normalization(layer, training=training)
                layer = self.leaky_relu(layer)
                layer = tf.layers.conv2d(layer, 256, 3, padding='same', use_bias=False)
                layer = tf.layers.batch_normalization(layer, training=training)
                layer = self.leaky_relu(layer)
                c2_up = upsample(c2, 2)
                layer = tf.concat([layer, c2_up], axis=-1)
                layer = self.leaky_relu(layer)
                layer = tf.layers.conv2d(layer, 128, 1, padding='same', use_bias=False)
                layer = tf.layers.batch_normalization(layer, training=training)
                layer = self.leaky_relu(layer)
                layer = tf.layers.conv2d(layer, 256, 3, padding='same')
                c3 = self.leaky_relu(layer)
                pred3 = tf.layers.conv2d(c3, num_anchor * (num_class + 5), 1, 1, activation=None)

            return pred1,pred2, pred3


    def pred_process(self, pred, cell_size, num_class, num_anchor):
        prediction = tf.reshape(pred, shape=[-1, cell_size, cell_size, num_anchor, num_class + 5])
        cla_pred = prediction[:, :, :, :, 0:num_class]
        conf_pred = prediction[:, :, :, :, num_class]
        loc_pred = prediction[:, :, :, :, num_class + 1:]
        return cla_pred, conf_pred, loc_pred

    def loc_process(self, loc_pred, cell_size, anchor, x_offset, y_offset):
        with tf.name_scope('loc_process'):
            xy_pred = loc_pred[:, :, :, :, :2]
            wh_pred = loc_pred[:, :, :, :, 2:]
            xy_activate = tf.nn.sigmoid(xy_pred)
            w_trans = tf.minimum(tf.exp(wh_pred[:, :, :, :, 0]) * anchor[:, :, :, :, 1] / cell_size, 1e+10)
            h_trans = tf.minimum(tf.exp(wh_pred[:, :, :, :, 1]) * anchor[:, :, :, :, 0] / cell_size, 1e+10)
            box_trans = tf.stack([(xy_activate[:, :, :, :, 0] + x_offset) / cell_size,
                                  (xy_activate[:, :, :, :, 1] + y_offset) / cell_size,
                                  w_trans,
                                  h_trans], axis=-1)
            return box_trans

    def loss(self, cla, conf, loc, labels, cell_size, anchor, offsets, num_class, batch_size):
        with tf.name_scope('loss'):
            cla_label = labels[:, :, :, :, 0:num_class]
            # print('**',cla_label.shape)
            center_response = labels[:, :, :, :, num_class]  # [0,13,13,5]
            loc_label = labels[:, :, :, :, num_class + 1:]
            x_off, y_off = offsets

            pred_box_trans = self.loc_process(loc, cell_size, anchor, x_off, y_off)

            iou = cal_iou(pred_box_trans, loc_label)
            max_iou = tf.reduce_max(iou, axis=3, keep_dims=True)
            # iou_max = tf.cast(iou >= max_iou, tf.float32) * iou  # ioumask shape [0,13,13,5]
            iou_mask = tf.cast(max_iou >= iou_threshold, tf.float32) * center_response

            noobject_iou = (tf.ones_like(iou_mask) - iou_mask)*iou
            noobj_mask = tf.cast(noobject_iou < iou_threshold, tf.float32)

            cla_loss = self.CLASS_SCALE * tf.reduce_sum(
                tf.nn.softmax_cross_entropy_with_logits(logits=cla, labels=cla_label,
                                                        dim=4) * center_response) / batch_size
            #
            tf.summary.scalar('cla_loss', cla_loss)
            # obj_loss = tf.reduce_sum(
            #     tf.nn.sigmoid_cross_entropy_with_logits(logits=conf, labels=iou_mask) * iou_mask) / batch_size

            obj_loss = self.OBJECT_SCALE * tf.reduce_sum(
                tf.square(tf.nn.sigmoid(conf) - iou_mask) * iou_mask) / batch_size

            tf.summary.scalar('obj_loss', obj_loss)

            no_obj_loss = self.NOOBJECT_SCALE * tf.reduce_sum(
                tf.square(tf.nn.sigmoid(conf) - 0) * noobj_mask) / batch_size

            tf.summary.scalar('no_obj_loss', no_obj_loss)

            x_loss = tf.reduce_sum(
                tf.square(pred_box_trans[:, :, :, :, 0] - loc_label[:, :, :, :, 0]) * center_response
            ) / batch_size
            y_loss = tf.reduce_sum(
                tf.square(pred_box_trans[:, :, :, :, 1] - loc_label[:, :, :, :, 1]) * center_response
            ) / batch_size
            w_loss = tf.reduce_sum(
                tf.square(tf.sqrt(pred_box_trans[:, :, :, :, 2]) - tf.sqrt(loc_label[:, :, :, :, 2])) * center_response
            ) / batch_size
            h_loss = tf.reduce_sum(
                tf.square(tf.sqrt(pred_box_trans[:, :, :, :, 3]) - tf.sqrt(loc_label[:, :, :, :, 3])) * center_response
            ) / batch_size
            tf.summary.scalar('x_loss', x_loss)
            tf.summary.scalar('y_loss', y_loss)
            tf.summary.scalar('w_loss', w_loss)
            tf.summary.scalar('h_loss', h_loss)

            loc_loss = self.COORD_SCALE * tf.add_n([x_loss, y_loss, w_loss, h_loss])
            tf.summary.scalar('loc_loss', loc_loss)

            all_loss = tf.add_n([cla_loss, obj_loss, no_obj_loss, loc_loss])

            return all_loss

    def leaky_relu(self, layer, alpha=0.1):
        return tf.maximum(layer, alpha * layer)

    def load_param(self, name, dict):
        return dict[name]

    def conv(self, layer, name, dict, pretrained=True, trainable=False):
        params = dict[name]
        weight = params[0]
        bias = params[1]
        if pretrained:
            weight = tf.Variable(weight, trainable=trainable, name=name + '_weights')
            bias = tf.Variable(bias, trainable=trainable, name=name + '_bias')
        if pretrained == False:
            weight = tf.get_variable(name=name + '_weights', shape=weight.shape)
            bias = tf.get_variable(name=name + '_bias', shape=bias.shape)
        return tf.nn.relu(
            tf.nn.conv2d(layer, weight, [1, 1, 1, 1], padding='SAME') + bias, name=name + 'active'
        )


if __name__ == '__main__':
    pass
