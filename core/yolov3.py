import tensorflow as tf
import numpy as np


import core.common as common
import core.backbone as backbone
from core.config import cfg
import core.utils as utils


NUM_CLASS = len(utils.read_class_names(cfg.YOLO.CLASSES))
ANCHORS = utils.get_anchors(cfg.YOLO.ANCHORS)
STRIDES = np.array(cfg.YOLO.STRIDES)
IOU_LOSS_THRESH = cfg.YOLO.IOU_LOSS_THRESHOLD


def YOLOv3(input_layer):
    """
    获得三种尺度上的输出
    """
    route1, route2, conv = backbone.darknet53(input_layer)

    conv = common.convolutional_set(conv, 1024, 1024, 512)
    conv_lbbox_branch = common.convolutional(conv, (3, 3, 512, 1024))
    conv_lbbox = common.convolutional(conv_lbbox_branch, (1, 1, 1024, 3 * (5 + NUM_CLASS)), activation=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 512, 256))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route2], axis=-1)

    conv = common.convolutional_set(conv, 768, 512, 256)
    conv_mbbox_branch = common.convolutional(conv, (3, 3, 256, 512))
    conv_mbbox = common.convolutional(conv_mbbox_branch, (1, 1, 512, 3 * (5 + NUM_CLASS)), activation=False, bn=False)

    conv = common.convolutional(conv, (1, 1, 256, 128))
    conv = common.upsample(conv)
    conv = tf.concat([conv, route1], axis=-1)

    conv = common.convolutional_set(conv, 384, 256, 128)
    conv_sbbox_branch = common.convolutional(conv, (3, 3, 128, 256))
    conv_sbbox = common.convolutional(conv_sbbox_branch, (1, 1, 256, 3 * (5 + NUM_CLASS)), activation=False, bn=False)

    # return conv_lbbox, conv_mbbox, conv_sbbox
    return [conv_sbbox, conv_mbbox, conv_lbbox]



def decode(conv_bbox, i):
    """
    将网络的输出logits转为网络输入尺寸下bbox坐标，其中网络输出有三种scale，一次处理一个scale
    [batch_size, output_size, output_size, 3 * (5 + 20)]
    """
    
    # 这里要用tf.shape!!!!!!!!!否则会报错：Failed to convert object of type <class 'tuple'> to Tensor.
    # 在1.0+版本中也有类似的问题，顺便注意在此检查common.residual_block
    # https://www.cnblogs.com/japyc180717/p/9321694.html
    # https://blog.csdn.net/yideqianfenzhiyi/article/details/79464725
    
    # batch_size = conv_bbox.shape[0]
    # output_size = conv_bbox.shape[1]

    conv_shape = tf.shape(conv_bbox)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    anchor = ANCHORS[i]  # (3, 2)
    stride = STRIDES[i]

    conv_bbox = tf.reshape(conv_bbox, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    # 5 --> x, y, w, h, conf
    x = tf.range(output_size)[tf.newaxis, :]  # shape = (1, output_size)
    y = tf.range(output_size)[:, tf.newaxis]  # shape = (output_size, 1)
    x_tile = tf.tile(x, [output_size, 1])[:, :, tf.newaxis]
    y_tile = tf.tile(y, [1, output_size])[:, :, tf.newaxis]  # shape = (output_size, output_size, 1)
    xy_tile = tf.concat((x_tile, y_tile), axis=-1)  # shape = (output_size, output_size, 2)
    xy_tile = xy_tile[tf.newaxis, :, :, tf.newaxis, :]  # shape = (1, output_size, output_size, 1, 2)
    xy_tile = tf.tile(xy_tile, [batch_size, 1, 1, 3, 1])  # shape = (batch_size, output_size, output_size, 3, 2)
    xy_tile = tf.cast(xy_tile, tf.float32)

    conv_dxdy = conv_bbox[..., 0:2]
    conv_dwdh = conv_bbox[..., 2:4]
    conv_conf = conv_bbox[..., 4:5]
    conv_prob = conv_bbox[..., 5:]

    pred_xy = (tf.sigmoid(conv_dxdy) + xy_tile) * stride
    # 最后两个维度对应元素相乘 shape不变， anchor定义在future map尺寸上
    pred_wh = (tf.exp(conv_dwdh) * anchor) * stride
    pred_conf = tf.sigmoid(conv_conf)
    pred_prob = tf.sigmoid(conv_prob)

    pred_bbox = tf.concat((pred_xy, pred_wh, pred_conf, pred_prob), axis=-1)
    return pred_bbox

"""
将pred_bbox与gt_bbox做iou
gt_bbox通过解析data得到，并扩展到pred_bbox的维度
"""
def bbox_iou(pred_bboxes, gt_bboxes):
    pred_area = pred_bboxes[..., 2] * pred_bboxes[..., 3]
    gt_area = gt_bboxes[..., 2] * gt_bboxes[..., 3]

    # inter section
    # x, y, w, h --> x_min, y_min, x_max, y_max
    bboxes1 = tf.concat((pred_bboxes[..., 0:2] - pred_bboxes[..., 2:4] * 0.5,
                         pred_bboxes[..., 0:2] + pred_bboxes[..., 2:4] * 0.5), axis=-1)

    bboxes2 = tf.concat((gt_bboxes[..., 0:2] - gt_bboxes[..., 2:4] * 0.5,
                         gt_bboxes[..., 0:2] + gt_bboxes[..., 2:4] * 0.5), axis=-1)
    left_up = tf.maximum(bboxes1[..., 0:2], bboxes2[..., 0:2])
    right_down = tf.minimum(bboxes1[..., 2:4], bboxes2[..., 2:4])
    # inter_section的w和h，防止right_down < left_up
    inter_section = tf.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    # union
    union_area = pred_area + gt_area - inter_area
    iou = 1.0 * inter_area / union_area
    return union_area, iou



def bbox_giou(pred_bboxes, gt_bboxes):
    union_area, iou = bbox_iou(pred_bboxes, gt_bboxes)

    # enclose_area
    bboxes1 = tf.concat((pred_bboxes[..., 0:2] - pred_bboxes[..., 2:4] * 0.5,
                         pred_bboxes[..., 0:2] + pred_bboxes[..., 2:4] * 0.5), axis=-1)
    bboxes2 = tf.concat((gt_bboxes[..., 0:2] - gt_bboxes[..., 2:4] * 0.5,
                         gt_bboxes[..., 0:2] + gt_bboxes[..., 2:4] * 0.5), axis=-1)
    left_up = tf.minimum(bboxes1[..., 0:2], bboxes2[..., 0:2])
    right_down = tf.maximum(bboxes1[..., 2:4], bboxes2[..., 2:4])
    enclose_section = tf.maximum(right_down - left_up , 0.0)
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]
    giou = iou - (enclose_area - union_area) / enclose_area
    return giou  # shape = (..., )


"""
计算pred_bbox和gt_bbox的loss
loss：giou_loss, conf_loss, prob_loss
"""
def compute_loss(pred_bboxes, conv, label, gt_bboxes, i):

    conv_shape = tf.shape(conv)
    batch_size = conv_shape[0]
    output_size = conv_shape[1]
    input_size = STRIDES[i] * output_size
    conv = tf.reshape(conv, (batch_size, output_size, output_size, 3, 5 + NUM_CLASS))

    conv_raw_conf = conv[..., 4:5]
    conv_raw_prob = conv[..., 5:]

    pred_xywh = pred_bboxes[..., 0:4]
    pred_conf = pred_bboxes[..., 4]

    label_xywh = label[..., 0:4]
    # # 特别注意：这里label_conf.shape = (batch_size, output_size, output_size, 3, 1)
    # # 如果是 label_conf = label[..., 4]  那么shape = (batch_size, output_size, output_size, 3)
    label_conf = label[..., 4:5]
    label_prob = label[..., 5:]

    """
    高维数组做乘积还是有问题
    到底要不要在最后加上第4维度？？ 只有label_的维度是4维才能解释
    """
    # giou_loss
    # output_shape = (batch, output_size, output_size, anchor_per_scale, 1)
    giou = tf.expand_dims(bbox_giou(pred_xywh, label_xywh), axis=-1)  # shape = (batch, output_size, output_size, 3, 1)
    input_size = tf.cast(input_size, tf.float32)
    bbox_loss_scale = 2.0 - 1.0 * label_xywh[..., 2] * label_xywh[..., 3] / (input_size ** 2)
    # # 有物体才计入损失
    giou_loss = label_conf * bbox_loss_scale * (1 - giou)

    # conf_loss
    # 1、background_conf
    # # pred_xywh.shape = (batch_size, output_size, output_size, anchor_per_scale, 4)
    # # gt_bboxes.shape = (batch_size, max_bbox_per_scale, 4)  原来有三个scale，通过train的i只选出一个
    # # iou.shape       = (batch_size, output_size, output_size, anchor_per_scale, max_bbox_per_scale) 会减掉最后一个维度
    _, iou = bbox_iou(pred_xywh[:, :, :, :, tf.newaxis, :], gt_bboxes[:, tf.newaxis, tf.newaxis, tf.newaxis, :, :])
    # # shape = (batch_size, output_size, output_size, anchor_per_scale, 1)
    max_iou = tf.expand_dims(tf.reduce_max(iou, axis=-1), axis=-1)
    iou_mask = max_iou < IOU_LOSS_THRESH
    # # shape见上方特别注意
    background_conf = (1 - label_conf) * tf.cast(iou_mask, tf.float32)

    # 2、facol factor
    conf_facol = tf.pow(label_conf - pred_conf, 2) * 0.25  # paper 中的alpha

    conf_loss = conf_facol * (
        label_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_conf, logits=conv_raw_conf)
        +
        background_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_conf, logits=conv_raw_conf)
    )

    # prob_loss
    prob_loss = label_conf * tf.nn.sigmoid_cross_entropy_with_logits(labels=label_prob, logits=conv_raw_prob)

    # vector --> scale
    giou_loss = tf.reduce_mean(tf.reduce_sum(giou_loss, axis=[1, 2, 3, 4]))
    conf_loss = tf.reduce_mean(tf.reduce_sum(conf_loss, axis=[1, 2, 3, 4]))
    prob_loss = tf.reduce_mean(tf.reduce_sum(prob_loss, axis=[1, 2, 3, 4]))

    return giou_loss, conf_loss, prob_loss

