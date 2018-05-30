#!/usr/bin/env python
# -*- coding:utf-8 -*-

import xml.etree.ElementTree as ET
import os
import cv2
import numpy as np
from util.cfg import cell_size,CLASS_NUM,ANCHOR_NUM
VGG_MEAN = [103.939, 116.779, 123.68]

CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
           'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
           'train', 'tvmonitor']

pascal_path = 'C:\VOC2012\VOCdevkit\VOC2012'
img_dir ='C:\VOC2012\VOCdevkit\VOC2012\JPEGImages'
data_path = os.path.join(pascal_path,'Annotations')
test_path = os.path.join(data_path, '2007_000027.xml')


def parse_xml(path):
    parses = ET.parse(path)
    root = parses.getroot()
    filename = root.find('filename').text.strip()
    sizes = root.find('size')
    width = int(sizes.find('width').text.strip())
    height = int(sizes.find('height').text.strip())
    objs = root.findall('object')
    labels = []
    for obj in objs:
        class_name = obj.find('name').text.strip()
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text.strip()))
        xmax = int(float(bbox.find('xmax').text.strip()))
        ymin = int(float(bbox.find('ymin').text.strip()))
        ymax = int(float(bbox.find('ymax').text.strip()))
        labels.append([(xmax + xmin)/(2.*width),(ymax + ymin)/(2.*height),
                       (xmax - xmin) / (1. * width), (ymax - ymin) / (1. *height) , CLASSES.index(class_name)])

    return os.path.join(img_dir, filename),labels

def labels_process(labels, cell_size):
    """
    :param labels:[[x,y,w,h,c],[x,y,w,h,c],[x,y,w,h,c]...]
    :return:
    """
    center_confirm = np.zeros([cell_size,cell_size,ANCHOR_NUM,CLASS_NUM+5])
    for label in labels:
        cx,cy,w,h = label[:4]
        cla = label[4]
        position_x = int(np.floor(cx*cell_size))
        position_y = int(np.floor(cy*cell_size))
        # abs_x = cx * cell_size- position_x
        # abs_y = cy * cell_size- position_y
        center_confirm[position_y,position_x,:,CLASS_NUM] = 1
        center_confirm[position_y, position_x, :, CLASS_NUM+1:] = [cx, cy, w, h]
        center_confirm[position_y, position_x, :, cla] = 1
    # center_confirm = np.tile(center_confirm,[1,1,5,1])
    return center_confirm

def load_batch(batch_size):
    data_list = os.listdir(data_path)
    n_batch = len(data_list) // batch_size
    abs_len = n_batch * batch_size
    data_list = data_list[:abs_len]
    data_list = np.array(data_list)
    data_list = list(data_list)
    np.random.shuffle(data_list)
    for i in range(n_batch):
        datas = []
        l1 = []
        l2 = []
        l3 = []
        batch_data = data_list[i*batch_size:(i+1)*batch_size]
        for path in batch_data:
            path = os.path.join(data_path, path)
            img_path, label = parse_xml(path)
            img = cv2.imread(img_path)
            img = cv2.resize(img,(416,416))
            img = img * 1.
            img = img - VGG_MEAN
            datas.append(img)
            l1.append(labels_process(label, cell_size[0]))
            l2.append(labels_process(label, cell_size[1]))
            l3.append(labels_process(label, cell_size[2]))
        yield np.array(datas), np.array(l1), np.array(l2), np.array(l3)


if __name__ == '__main__':
    data = load_batch(10)
    x,y1,y2,y3 = next(data)
    print(y3.shape)