from yolo.yolo3 import *
from util.cfg import *
from datas.data_generate import load_batch
# from util.process_voc import load_batch as load_voc
import time
from resnet.net import *
import cv2

class SOLVER(object):
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
        self.res = RESNET(self.dict,False)
        self.init()
        self.feat1, self.feat2, self.feat3 = self.res.feat
        self.dict = None

        self.offsets = []
        for i in range(3):
            self.offsets.append(offset(cell_size=self.CELL_SIZE[i]))

        self.preds = \
            self.net.head(self.feat1, self.feat2, self.feat3,self.CLASS_NUM,self.ANCHOR_NUM, self.training)

        self.clas, self.conf_pred, self.box = self.pred_process()

        self.predict()

    def init(self):
        with tf.name_scope('input_ph'):
            self.inputs = self.res.x
            self.training = tf.placeholder(tf.bool)


    def pred_prerocess_per(self, pred, cell_size, anchor, classnum, num_anchor, offsets):
        cla_pred, conf_pred, loc_pred = self.net.pred_process(pred, cell_size,classnum, num_anchor)
        box = self.net.loc_process(loc_pred, cell_size, anchor, offsets[0], offsets[1])
        return cla_pred, conf_pred, box

    def pred_process(self):
        cla_pred, conf_pred, box = [],[],[]
        for i in range(3):
            cla, conf, bx = self.pred_prerocess_per(self.preds[i],self.CELL_SIZE[i],self.ANCHORS[i],self.CLASS_NUM, self.ANCHOR_NUM, self.offsets[i])
            cla_pred.append(tf.reshape(cla, [-1, self.CLASS_NUM]))
            conf_pred.append(tf.reshape(tf.expand_dims(conf, -1), [-1, 1]))
            box.append(tf.reshape(bx, [-1, 4]))
        clas = tf.concat(cla_pred, axis=0)
        conf_pred = tf.concat(conf_pred, axis=0)
        box = tf.concat(box, axis=0)
        return clas, conf_pred, box


    def predict(self):
        self.conf_score = tf.nn.sigmoid(self.conf_pred)
        self.cla = tf.nn.softmax(self.clas)
        self.response = self.conf_score * self.cla
        # #####(1,13,13,5,20)
        #
        self.c = tf.argmax(self.response, axis=-1)
        self.scores = tf.reduce_max(self.response, axis=-1)
        self.mask = self.scores > 0.5
        self.filter_c = tf.boolean_mask(self.c, self.mask)
        self.filter_scores = tf.boolean_mask(self.scores, self.mask)
        self.filter_box = tf.boolean_mask(self.box, self.mask)

    def img_process(self, path):
        img = cv2.imread(path)
        img = cv2.resize(img, (image_size[0],image_size[1]))
        cimg = np.copy(img)
        cimg = cimg *1.
        cimg = cimg - VGG_MEAN
        cimg = np.reshape(cimg, [1]+image_size)
        return img, cimg

    def video_process(self, img):
        img = cv2.resize(img, (image_size[0], image_size[1]))
        cimg = np.copy(img)
        cimg = cimg * 1.
        cimg = cimg - VGG_MEAN
        cimg = np.reshape(cimg, [1] + image_size)
        return img, cimg

    def test(self, path):
        restore = tf.train.Saver()
        t1 = time.time()
        print('RESTORE')
        restore.restore(self.sess, save_path)
        t2 = time.time()
        print('TAKES %f SECONDS RESTORE FILE FROM : '%(t2-t1),save_path)
        img, cimg = self.img_process(path)
        img = cv2.resize(img, (600,600))
        cs, confs, bxs = self.sess.run([self.filter_c, self.filter_scores, self.filter_box], feed_dict={self.inputs:cimg, self.training:False})
        cs, confs, bxs = box_filter(cs, confs, bxs )
        img = draw_result(img, cs,confs,bxs)
        cv2.imshow('w', img)
        cv2.waitKey()

    def detect_capture(self):
        restore = tf.train.Saver()
        t1 = time.time()
        print('RESTORE')
        restore.restore(self.sess, save_path)
        t2 = time.time()
        print('TAKES %f SECONDS RESTORE FILE FROM : ' % (t2 - t1), save_path)
        cap = cv2.VideoCapture(0)
        success, frame = cap.read()
        while success:
            success, frame = cap.read()
            img, cimg = self.video_process(frame)
            cs, confs, bxs = self.sess.run([self.filter_c, self.filter_scores, self.filter_box],
                                           feed_dict={self.inputs: cimg, self.training: False})
            cs, confs, bxs = box_filter(cs, confs, bxs)
            img = draw_result(img, cs, confs, bxs)
            cv2.imshow('w', img)
            cv2.waitKey(1)



if __name__ == '__main__':
    solver = SOLVER()
    # print(solver.filter_c.shape, solver.conf_pred.shape, solver.box.shape)
    # solver.test('./image/test3.jpg')
    # solver.detect_capture()




















