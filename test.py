# -*- coding: utf-8 -*-
"""
Created on Wed May 13 12:56:24 2020

@author: Administrator
"""

from core.yolov3 import YOLOv3, decode
import core.utils as utils
from core.config import cfg

import tensorflow as tf
import cv2
import numpy as np
import shutil
import os


# 确定参数

INPUT_SIZE = 416
CLASSES = utils.read_class_names(cfg.YOLO.CLASSES)
NUM_CLASS = len(CLASSES)

predicted_dir_path = '../mAP/predicted'
ground_truth_dir_path = '../mAP/ground_truth'
if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
if os.path.exists(cfg.TEST.DETECTED_IMAGE_PATH): shutil.rmtree(cfg.TEST.DETECTED_IMAGE_PATH)
os.mkdir(predicted_dir_path)
os.mkdir(ground_truth_dir_path)
os.mkdir(cfg.TEST.DETECTED_IMAGE_PATH)


# build model

input_layer = tf.keras.Input(shape = [INPUT_SIZE, INPUT_SIZE, 3])
feature_maps = YOLOv3(input_layer)
output_layer = []
for i, fm in enumerate(feature_maps):
    output_tensor = decode(fm, i)
    output_layer.append(output_tensor)
    
model = tf.Model(inputs = input_layer, outputs = output_layer)
model.load_weights('./yolov3')


# 打开test数据文件，边测边写
with open(cfg.TEST.ANNOT_PATH, 'r') as annotation_file:
    # annotation = annotation_file.readlines()
    for num, line in enumerate(annotation_file):
        annotation = line.strip().split()
        image_path = annotation[0]
        image_name = annotation.split('/')[-1]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # [ for i ] for前是对i的全部操作，[]的结果就是操作玩的所有结果
        bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])
        
        # 注意没有gt_bbox的情况
        # if bbox_data_gt is None:
        if len(bbox_data_gt) == 0:
            bboxes_gt = []
            classes_gt = []
        else:
            bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
        ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')
        
        print('==> ground truth of %s:' % image_name)
        num_bbox_gt = len(bboxes_gt)
        # 将gt_bbox信息写入文件
        with open(ground_truth_path, 'w') as f:
            for i in range(bboxes_gt):
                class_name = CLASSES[classes_gt[i]]
                xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))  # 写文件 --> str
                gt_bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                f.write(gt_bbox_mess)
                print('\t' + str(gt_bbox_mess).strip())
                
        # predict process
        print('predict result of %s:' % image_name)
        predicted_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
        
        image_size = image.shape[:2]
        image_data = utils.image_preprocess(np.copy(image), INPUT_SIZE)  # np.copy()
        image_data = image[np.newaxis, :, :]
        
        pred_bbox = model.predict(image_data)
        # 3 * 3 --> 3,
        pred_bbox = [tf.reshape(-1, (tf.shape(x)[-1])) for x in pred_bbox]
        pred_bbox = tf.concat(pred_bbox, axis = 0)
        bboxes = utils.postprocess_boxes(pred_bbox, image_size, INPUT_SIZE, cfg.TEST.SCORE_THRESHOLD)
        bboxes = utils.nms(bboxes, cfg.TEST.IOU_THRESHOLD, method='nms')
        
        
        # 图片写道路径
        if cfg.TEST.DETECTED_IMAGE_PATH is not None:
            image = utils.draw_bboxes(image, bboxes)
            cv2.imwrite(cfg.TEST.DETECTED_IMAGE_PATH + image_name, image)
            
        # 写入预测结果，带score
        with open(predicted_result_path, 'w') as f:
            for bbox in bboxes:
                # coor = bbox[:4]
                coor = np.array(bbox[:4], dtype = np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                class_name = CLASSES[class_ind]
                # int,float --> str
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = list(map(str, coor))
                pred_bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                f.write(pred_bbox_mess)
                print('\t' + str(pred_bbox_mess).strip())
            
        
























