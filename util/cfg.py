#!/usr/bin/env python
# -*- coding:utf-8 -*-
import numpy as np

VGG_FILE = 'D:/CODE5.24/vgg_yolo/paramaters/vgg.npy'

save_path = './save/pika.ckpt-9'
VGG_MEAN = [103.939, 116.779, 123.68]
image_size = [416,416,3]
CLASS_NUM = 20
ANCHOR_NUM = 3
cell_size = [13, 26, 52]
iou_threshold = 0.7
lr = 0.0001
batch_size = 4
num_epochs = 1000
# anchors = [
#             [1.81218901641, 2.0756480568],
#             [3.27746965391, 5.96042296557],
#             [4.84211551027, 8.96603606529],
#             [9.99847701672, 6.52768518575],
#             [11.101054447, 9.88710144146]
# ]
# anchors =  [[0.57273, 0.677385], [1.87446, 2.06253], [3.33843, 5.47434], [7.88282, 3.52778], [9.77052, 9.16828]]
# anchors = np.reshape(np.array(anchors),[1,1,1,5,2])
anchors1 = [[10,13],  [16,30],  [33,23]]
anchors2 = [[30,61],  [62,45],  [59,119]]
anchors3 = [[116,90],  [156,198],  [373,326]]
anchors = [np.reshape(np.array(anchors3),[1,1,1,3,2])/ 416.*13,
           np.reshape(np.array(anchors2),[1,1,1,3,2])/ 416.*26,
           np.reshape(np.array(anchors1),[1,1,1,3,2])/ 416.*52, ]

# print(anchors)

pretrain_class = 808
iou_lower = 0.3

def offset(cell_size=cell_size):
    y,x = np.mgrid[0:cell_size,0:cell_size]
    x = np.tile(np.reshape(x, [cell_size, cell_size, 1]), [1, 1, ANCHOR_NUM])
    y = np.tile(np.reshape(y, [cell_size, cell_size, 1]), [1, 1, ANCHOR_NUM])
    return x,y


# bias_match=1
# classes=80
# coords=4
# num=5
# softmax=1
# jitter=.3
# rescore=1
#
object_scale=5.
noobject_scale=0.5
class_scale=1.
coord_scale=1.