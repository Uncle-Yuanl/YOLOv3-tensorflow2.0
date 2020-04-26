import tensorflow as tf
import cv2
import numpy as np
from PIL import Image

from core.yolov3 import YOLOv3, decode
import core.utils as utils

"""
1、定义模型
"""
input_size = 416
image_path = "D:/Anacoda/YOLO_v3_s/docs/kite.jpg"
input_layer = tf.keras.layers.Input(shape=[input_size, input_size, 3])
conv_bboxes = YOLOv3(input_layer)
output_layers = []
for i, conv_bbox in enumerate(conv_bboxes):
    pred_bbox = decode(conv_bbox, i)
    output_layers.append(pred_bbox)

model = tf.keras.Model(inputs=input_layer, outputs=output_layers)
# 加载权重
utils.load_weights(model, r"D:\Anacoda\YOLO_v3_s\docs\yolov3.weights")
model.summary()

"""
2、读取测试图片
"""
original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_size = original_image.shape[:2]

image_data = utils.image_preprocess(np.copy(original_image), input_size)
image_data = image_data[np.newaxis, ...].astype(np.float32)

"""
输入网络，得出结果，并在original_image上显示
"""
pred_bboxes = model.predict(image_data)
pred_bboxes = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bboxes]  # 每个元素应该是(None, 85)
pred_bboxes = tf.concat(pred_bboxes, axis=0)
bboxes = utils.postprocess_boxes(pred_bboxes, original_size, input_size, 0.3)
bboxes = utils.nms(bboxes, 0.45, method='nms')

image = utils.draw_bboxes(original_image, bboxes)
image = Image.fromarray(image)
image.show()

