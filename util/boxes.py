#!/usr/bin/env python
# -*- coding:utf-8 -*-
import tensorflow as tf
import cv2
import numpy as np
import time
from util.process_voc import CLASSES
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

# def box_filter(boxes, conf):
#     results = []
#     for i in range(13):
#         for j in range(13):
#             for k in range(5):
#                 if conf[0,i,j,k] == 1.:
#                     box = boxes[0,i,j,k,:]
#                     print(box)
#                     results.append([1,box])
#     return results
# cccccccc = ['','pikachu']
def draw_result(img, cs ,confs,boxes):
    font = cv2.FONT_HERSHEY_SIMPLEX
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
        img = cv2.putText(img, CLASSES[cs[i]] + '%.2f'%confs[i], (xmin, ymin), font, 0.4, (0, 0, 255),1)
    return img


def iou_one(box1, box2):
    xmin1 = box1[0] - box1[2] / 2.
    ymin1 = box1[1] - box1[3] / 2.
    xmax1 = box1[0] + box1[2] / 2.
    ymax1 = box1[1] + box1[3] / 2.
    xmin2 = box2[0] - box2[2] / 2.
    ymin2 = box2[1] - box2[3] / 2.
    xmax2 = box2[0] + box2[2] / 2.
    ymax2 = box2[1] + box2[3] / 2.
    interw = np.maximum(0., np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2))
    interh = np.maximum(0., np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2))
    inter = interw*interh
    union = box1[2]*box1[3] + box2[2]*box2[3] - inter
    return np.clip(inter / union , 0 , 1)

def box_filter(cs, confs, bxs):
    t1 = time.time()
    confs_idx = list(np.argsort(confs))
    num = len(confs_idx)
    if num == 0:
        return np.array([]),np.array([]),np.array([])
    cccs = [cs[confs_idx[0]]]
    ccconfs = [confs[confs_idx[0]]]
    cccbox = [bxs[confs_idx[0]]]
    for i in range(1, num):
        is_app = True
        j = 0
        while j < len(cccbox):
            ious = iou_one(cccbox[j], list(bxs[confs_idx[i]]))
            c = cccs[j]
            j += 1
            if ious > 0.3 and c == cs[confs_idx[i]]:
                j = len(cccbox)
                is_app = False
        if is_app:
            cccs.append(cs[confs_idx[i]])
            ccconfs.append(confs[confs_idx[i]])
            cccbox.append(list(bxs[confs_idx[i]]))

    t2 = time.time()
    print('Filter %d boxes takes %f seconds outputs %d boxes'%(num, t2-t1, len(cccbox)))
    return np.array(cccs), np.array(ccconfs), np.array(cccbox)


if __name__ == '__main__':
    cc = np.random.uniform(0,10,[50])
    con = np.random.uniform(0,10,[50])
    bbx = np.random.uniform(0, 10, [50, 4])
    cccs, ccconfs, cccbox = box_filter(cc,con,bbx)
    print(cccs)


