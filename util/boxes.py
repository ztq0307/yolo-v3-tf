#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import cv2

def cal_iou(box1, box2):
    """
    :param box1:[batchsize, ceil, ceil, anchor, 4]
    :param box2:[batchsize, ceil, ceil, 1, 4]
    :return:
    """
    xmin1 = box1[:, :, :, :, 0] - box1[:, :, :, :, 2] / 2.
    ymin1 = box1[:, :, :, :, 1] - box1[:, :, :, :, 3] / 2.
    xmax1 = box1[:, :, :, :, 0] + box1[:, :, :, :, 2] / 2.
    ymax1 = box1[:, :, :, :, 1] + box1[:, :, :, :, 3] / 2.
    xmin2 = box2[:, :, :, :, 0] - box2[:, :, :, :, 2] / 2.
    ymin2 = box2[:, :, :, :, 1] - box2[:, :, :, :, 3] / 2.
    xmax2 = box2[:, :, :, :, 0] + box2[:, :, :, :, 2] / 2.
    ymax2 = box2[:, :, :, :, 1] + box2[:, :, :, :, 3] / 2.
    box1_trans = tf.stack([xmin1, ymin1, xmax1, ymax1], axis=-1)
    box2_trans = tf.stack([xmin2, ymin2, xmax2, ymax2], axis=-1)
    inter_w = tf.minimum(box1_trans[:, :, :, :, 2], box2_trans[:, :, :, :, 2]) - tf.maximum(box1_trans[:, :, :, :, 0],
                                                                                            box2_trans[:, :, :, :, 0])
    inter_h = tf.minimum(box1_trans[:, :, :, :, 3], box2_trans[:, :, :, :, 3]) - tf.maximum(box1_trans[:, :, :, :, 1],
                                                                                            box2_trans[:, :, :, :, 1])
    intersection = inter_w * inter_h

    square1 = box1[:, :, :, :, 2] * box1[:, :, :, :, 3]
    square2 = box2[:, :, :, :, 2] * box2[:, :, :, :, 3]
    union = square1 + square2 - intersection

    return tf.clip_by_value(intersection / union, 0, 1)

def box_filter(boxes, conf):
    results = []
    for i in range(13):
        for j in range(13):
            for k in range(5):
                if conf[0,i,j,k] == 1.:
                    box = boxes[0,i,j,k,:]
                    print(box)
                    results.append([1,box])
    return results

def draw_result(img, cs,confs,boxes):
    himg = img.shape[0]
    wimg = img.shape[1]
    num = cs.shape[0]
    for i in range(num):
        if cs[i] == 0:
            continue
        cx,cy,w,h = list(boxes[i])
        cx *= wimg
        cy *= himg
        w *= wimg
        h *= himg
        xmin = int(cx - w / 2)
        ymin = int(cy - h / 2)
        xmax = int(cx + w / 2)
        ymax = int(cy + h / 2)
        img = cv2.rectangle(img,(xmin, ymin),(xmax,ymax),(0,255,0),2)
    return img











