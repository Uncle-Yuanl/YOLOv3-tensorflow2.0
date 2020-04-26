import random
import colorsys
import numpy as np
import cv2

from core.config import cfg

def load_weights(model, weights_file):
    """
    直接搬的，我也不太会
    没有手动定义layer的名字，风险太大，在spyder中一次运行不成功
    默认layer的名字的后缀就会叠加，导致找不到layer，pycharm没事
    """
    wf = open(weights_file, 'rb')
    major, minor, revision, seen, _ = np.fromfile(wf, dtype=np.int32, count=5)

    j = 0
    for i in range(75):
        conv_layer_name = 'conv2d_%d' %i if i > 0 else 'conv2d'
        bn_layer_name = 'batch_normalization_%d' %j if j > 0 else 'batch_normalization'

        conv_layer = model.get_layer(conv_layer_name)
        filters = conv_layer.filters
        k_size = conv_layer.kernel_size[0]
        in_dim = conv_layer.input_shape[-1]

        if i not in [58, 66, 74]:
            # darknet weights: [beta, gamma, mean, variance]
            bn_weights = np.fromfile(wf, dtype=np.float32, count=4 * filters)
            # tf weights: [gamma, beta, mean, variance]
            bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
            bn_layer = model.get_layer(bn_layer_name)
            j += 1
        else:
            conv_bias = np.fromfile(wf, dtype=np.float32, count=filters)

        # darknet shape (out_dim, in_dim, height, width)
        conv_shape = (filters, in_dim, k_size, k_size)
        conv_weights = np.fromfile(wf, dtype=np.float32, count=np.product(conv_shape))
        # tf shape (height, width, in_dim, out_dim)
        conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

        if i not in [58, 66, 74]:
            conv_layer.set_weights([conv_weights])
            bn_layer.set_weights(bn_weights)
        else:
            conv_layer.set_weights([conv_weights, conv_bias])

    assert len(wf.read()) == 0, 'failed to read all data'
    wf.close()


def read_class_names(path):
    with open(path, 'r') as f:
        # data = ['person\n', 'bicycle\n',...]
        data = f.readlines()
        # data = ['person', 'bicycle',...]
        data = [x.strip('\n') for x in data]
    return data

def get_anchors(path):
    with open(path, 'r') as f:
        anchors_str = f.readline()
        anchors = np.array(anchors_str.split(','), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


def image_preprocess(image, input_size, gt_bboxes = None):
    """
    缩放至网络输入的尺寸
    """
    h, w = image.shape[:2]
    scale = min(input_size / h, input_size / w)
    nh, nw = int(h * scale), int(w * scale)

    image_resized = cv2.resize(image, (nw, nh))  # integer argument expected, got float
    image_padded = np.full(shape=[input_size, input_size, 3], fill_value=128.0)
    # resized图像放在中间
    dw, dh = (input_size - nw) // 2, (input_size - nh) // 2
    image_padded[dh:dh+nh, dw:dw+nw, :] = image_resized
    image_padded = image_padded / 255.

    # 如果训练数据，有gt_bboxes，相应位置大小调整
    # x_min, y_min, x_max, y_max
    if gt_bboxes:
        gt_bboxes[..., [0, 2]] = gt_bboxes[..., [0, 2]] * scale + dw
        gt_bboxes[..., [1, 3]] = gt_bboxes[..., [1, 2]] * scale + dh
        return image_padded, gt_bboxes
    return image_padded


def postprocess_boxes(pred_bboxes, original_size, input_size, score_thresh):
    """
    1、将网络预测出三种scale下的许多conv_bboxes
    2、通过decode转为了input_size上的pred_bboxes，
    3、再转到原始图片尺寸上，并去掉一些低效框
    4、保留 coor（x_min_org, y_min_org, x_max_org, y_max_org） score以及class_id 
    """
    org_h, org_w = original_size
    pred_bboxes = np.array(pred_bboxes)  # tf --> np

    pred_xywh = pred_bboxes[:, 0:4]
    pred_conf = pred_bboxes[:, 4]
    pred_prob = pred_bboxes[:, 5:]

    # x, y, w, h --> x_min, y_min, y_max
    pred_coor = np.concatenate((pred_xywh[:, 0:2] - pred_xywh[:, 2:4] * 0.5,
                                pred_xywh[:, 0:2] + pred_xywh[:, 2:4] * 0.5), axis=-1)
    # input_size --> original_size
    # 按照image_preprocess反着处理
    org_h, org_w = original_size
    resize_ratio = min(input_size / org_w, input_size / org_h)

    dw = (input_size - resize_ratio * org_w) / 2
    dh = (input_size - resize_ratio * org_h) / 2

    pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
    pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

    # 剔掉无效框，min > max
    pred_coor = np.concatenate((np.maximum(pred_coor[:, 0:2], [0, 0]),
                                np.minimum(pred_coor[:, 2:4], [org_w - 1, org_h - 1])), axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]),
                                 (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0

    # 剔掉无效框, 超出尺寸 感觉跟上面的一个意思
    valid_score = [0, np.inf]
    # 面积开方
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_score[0] < bboxes_scale), (bboxes_scale < valid_score[1]))

    # 剔掉低分框
    class_id = np.argmax(pred_prob, axis=-1)  # (None, )
    scores = pred_conf * pred_prob[np.arange(len(pred_prob)), class_id]
    score_mask = scores > score_thresh
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], class_id[mask]

    bboxes = np.concatenate((coors, scores[:, np.newaxis], classes[:, np.newaxis]), axis=-1)
    return bboxes


def bboxes_iou(bboxes1, bboxes2):
    """
    bboxes：(None, 6)  coors: x_min, y_min, x_max, y_max
    """
    bboxes1 = np.array(bboxes1)
    bboxes2 = np.array(bboxes2)

    bboxes1_area = (bboxes1[..., 2] - bboxes1[..., 0]) * (bboxes1[..., 3] - bboxes1[..., 1])
    bboxes2_area = (bboxes2[..., 2] - bboxes2[..., 0]) * (bboxes2[..., 3] - bboxes2[..., 1])

    left_up = np.maximum(bboxes1[..., 0:2], bboxes2[..., 0:2])
    right_down = np.minimum(bboxes1[..., 2:4], bboxes2[..., 2:4])
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]

    union_area = bboxes1_area + bboxes2_area - inter_area
    # # 避免iou过小 1.1920929e-07
    ious = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    bboxes 都是再原始尺寸上的bboxes
    bboxes : (None, 6)  coors + score + class
    预测出同类别物体的框一块处理，先选出score最高的框，与剩下的框做iou
    大于threshold --> 同一个物体重叠度过高，冗余框
    小于threshold --> 同一个类别，远处的其他物体
    因此threshold也控制了检测重叠目标的能力
    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    # 提取出预测框中可能的所有类别
    classes_in_image = list(set(bboxes[:, 5]))
    best_bboxes = []
    # 同类的一起处理
    for cls in classes_in_image:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_score_id = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_score_id]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate((cls_bboxes[0:max_score_id], cls_bboxes[max_score_id + 1:]))

            ious = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            # 不直接用mask是因为还要考虑到soft-nms
            weights = np.ones((len(ious),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']
            if method == 'nms':
                weights[ious > iou_threshold] = 0.0
            if method == 'soft-nms':
                weights = np.exp(-(1.0 * ious ** 2 / sigma))

            # 通过score来筛选bboxes
            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weights
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def draw_bboxes(image, bboxes, classes=read_class_names(cfg.YOLO.CLASSES), show_label=True):
    """
    bboxes: (x_min, y_min, x_max, y_max, score, class_id)
    """
    num_classes = len(classes)  # image_demo中可以传过来21个框，每个(82, )
    image_h, image_w, _ = image.shape
    # 定义颜色
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(0)
    random.shuffle(colors)
    random.seed(None)

    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[0:4], dtype=np.int32)  # 这里是int32了，cv2.rectangle中需要
        score = bbox[4]
        cls_id = int(bbox[5])
        bbox_color = colors[cls_id]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            fontScale = 0.5
            bbox_msg = "%s: %.2f" % (classes[cls_id], score)
            t_size = cv2.getTextSize(bbox_msg, 0, fontScale, thickness=bbox_thick // 2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled
            cv2.putText(image, bbox_msg, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    return image






