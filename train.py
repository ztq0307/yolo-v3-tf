#!/usr/bin/env python
# -*- coding:utf-8 -*-
from yolo.yolo3 import *
from util.cfg import *
from datas.data_generate import load_batch
from util.process_voc import load_batch as load_voc
import time
from resnet.net import *

class SOVLER(object):
    def __init__(self):
        self.sess = tf.InteractiveSession()
        self.IMAGE_SIZE = [None] + image_size
        self.ANCHOR_NUM = ANCHOR_NUM
        self.CLASS_NUM = CLASS_NUM
        self.ANCHORS = anchors
        self.CELL_SIZE = cell_size
        self.BATCHSIZE = batch_size
        self.LEARNING_RATE = lr
        self.net = YOLO2()
        self.dict = np.load('D:\CODE5.24\ResYOLO\\resnet\\reskey.npy').item()
        self.res = RESNET(self.dict,training=True)
        self.init()
        self.feat1, self.feat2, self.feat3 = self.res.feat
        self.dict = None

        self.offsets = []
        for i in range(3):
            self.offsets.append(offset(cell_size=self.CELL_SIZE[i]))

        self.preds = \
            self.net.head(self.feat1, self.feat2, self.feat3,self.CLASS_NUM,self.ANCHOR_NUM, self.training)

        # self.yolo_cost = self.net.loss(self.clas, self.confs, self.locs, self.labels, self.CELL_SIZE, self.CLASS_NUM, self.BATCHSIZE)
        self.cost_init()

        self.global_step = tf.Variable(0, trainable=False)
        self.lr = tf.train.exponential_decay(lr, self.global_step, 1000, 0.90)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.opt = (
                tf.train.AdamOptimizer(self.lr).minimize(self.cost, global_step=self.global_step)
            )

        self.vars = tf.global_variables()
        self.resvars = self.res.var
        self.save_res = tf.train.Saver(var_list=self.resvars)
        self.restore()

    def restore(self):
        self.save_res.restore(self.sess, './ResWeight/ResNet-L50.ckpt')

    def init(self):
        with tf.name_scope('input_ph'):
            self.inputs = self.res.x
            self.label1 = tf.placeholder(tf.float32,[None,self.CELL_SIZE[0],self.CELL_SIZE[0],self.ANCHOR_NUM,self.CLASS_NUM+5])
            self.label2 = tf.placeholder(tf.float32,[None,self.CELL_SIZE[1],self.CELL_SIZE[1],self.ANCHOR_NUM,self.CLASS_NUM+5])
            self.label3 = tf.placeholder(tf.float32,[None,self.CELL_SIZE[2],self.CELL_SIZE[2],self.ANCHOR_NUM,self.CLASS_NUM+5])
            self.labels = [self.label1, self.label2, self.label3]
            self.training = tf.placeholder(tf.bool)

    def cost_init(self):
        self.cost = 0
        for i in range(3):
            cla, conf, loc = self.net.pred_process(self.preds[i], self.CELL_SIZE[i], self.CLASS_NUM, self.ANCHOR_NUM)
            self.cost += self.net.loss(cla, conf, loc, self.labels[i],self.CELL_SIZE[i],self.ANCHORS[i], self.offsets[i], self.CLASS_NUM, self.BATCHSIZE, )

    def train(self, num=num_epochs, is_pretrain=False):
        saver = tf.train.Saver()
        vars = tf.global_variables()
        initvars = [var for var in vars if var not in self.resvars]
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('./graph/', self.sess.graph)
        if is_pretrain:
            print('Restore from :',save_path)
            ts = time.time()
            saver.restore(self.sess, save_path)
            te = time.time()
            print('OVER restore %f seconds'%(te-ts))
        if is_pretrain == False:
            print('START TRAINING ')
            self.sess.run(tf.variables_initializer(initvars))
        step = 0
        for i in range(1,21):
            # datas = load_batch(self.BATCHSIZE)
            datas = load_voc(self.BATCHSIZE)
            if i % 3 == 0:
                saver.save(self.sess, save_path ,global_step=i)
            for x, y1,y2,y3 in datas:
                step += 1
                feed = {self.inputs: x, self.label1: y1,self.label2: y2,self.label3: y3, self.training: True}
                self.sess.run(self.opt, feed_dict=feed)
                if step % 20 == 0:
                    feed_t = {self.inputs: x, self.label1: y1,self.label2: y2,self.label3: y3, self.training: False}
                    merge, loss = self.sess.run([merged, self.cost], feed_dict=feed_t)
                    writer.add_summary(merge, global_step=step)
                    print('Epoch %d after %d loss is %f ' % (i, step, loss))
        saver.save(self.sess, save_path, global_step=1)


if __name__ == '__main__':
    solver = SOVLER()
    solver.train()