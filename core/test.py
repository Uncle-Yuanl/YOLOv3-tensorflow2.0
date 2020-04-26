
import tensorflow as tf
import numpy as np
'''
"""
concat
"""
a = tf.constant([[[1, 2, 3], [1, 2, 3]], [[1, 2, 3], [1, 2, 3]]])
b = tf.constant([[[1, 1], [2, 2]], [[1, 1], [2, 2]]])
print(a)
print(b)

print(tf.concat((a, b), axis=-1))

"""
tile
"""
output_size = 10
x = tf.range(output_size)[tf.newaxis, :]  # shape = (1, output_size)
y = tf.range(output_size)[:, tf.newaxis]  # shape = (output_size, 1)
x_tile = tf.tile(x, [output_size, 1])[:, :, tf.newaxis]
y_tile = tf.tile(y, [1, output_size])[:, :, tf.newaxis]  # shape = (output_size, output_size, 1)
xy_tile = tf.concat((x_tile, y_tile), axis=-1)  # shape = (output_size, output_size, 2)
print(xy_tile)

"""
tf.newaxis
"""
a = tf.constant([[1, 2], [3, 4], [5, 6]])[tf.newaxis, ...]
b = tf.constant([[2, 2], [2, 2], [2, 2]])[tf.newaxis, ...]
b1 = tf.tile(b, [2, 1, 1])
print(a * b)
print(a * b1)


a = tf.constant([[1, 2], [3, 4], [5, 6]])[tf.newaxis, ...]
print(tf.shape(a))
print(a.shape)
print(tf.shape(a)[1], a.shape[1])


"""
sigmoid_cross_entropy_with_logits
"""
conv = tf.constant([[1., 2., 3.], [4., 5., 6.], [4., 2., 1.]])
pred = tf.nn.sigmoid(conv)   # 概率
label = tf.constant([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

scel = tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=conv)
cace = tf.losses.categorical_crossentropy(label, pred)
bce = tf.losses.binary_crossentropy(label, pred)

print(pred)

print("sigmoid_cross_entropy_with_logits: \n", scel)
print("reduce_mean: \n", tf.reduce_mean(scel, axis=1))
# print(tf.reduce_mean(scel, axis=1)) # 不等
print("categorical_crossentropy: \n", cace)
print("binary_crossentropy: \n", bce)

labeld = tf.constant([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
                     [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]])
predd = tf.sigmoid(tf.constant([[[1., 2., 3.], [4., 5., 6.], [4., 2., 1.]],
                                [[1., 2., 3.], [4., 5., 6.], [4., 2., 1.]]]))
print(tf.losses.binary_crossentropy(labeld, predd))



"""
iou dims
(2, 4)     --> (2, )
(2, 2, 4)  --> (2, 2)
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


pred_bboxes = tf.constant([[[5., 5., 2., 2.], [7., 7., 4., 4.]],
                           [[5., 5., 2., 2.], [7., 7., 4., 4.]]])
gt_bboxes = tf.constant([[[5., 5., 4., 4.], [7., 7., 2., 2.]],
                        [[5., 5., 4., 4.], [7., 7., 2., 2.]]])
union_area, iou = bbox_iou(pred_bboxes, gt_bboxes)
print(union_area)
print(iou)


"""
高维数组 *
"""
pred_bboxes = tf.constant([[[5., 5., 2., 2.], [7., 7., 4., 4.]],
                           [[5., 5., 2., 2.], [7., 7., 4., 4.]]])

a = pred_bboxes[..., 1]
aa = pred_bboxes[..., 1:2]
b = tf.reshape(a, (2, 2, 1))
print(a)
print(aa)
print(b)
print(a * b)


c = tf.reshape(tf.range(12), (2, 2, 3,))
d = tf.reshape(tf.range(12), (2, 2, 3, 1))
print(c)
print(c < 5)
print(c * d)


"""
np.multiply.reduce
"""
a = np.arange(1, 9).reshape(2, 2, 2)
print("a:", a)
b = np.multiply.reduce(a, axis=0)
print("b:", b)
c = np.multiply.reduce(a, axis=1)
print("c:", c)
d = np.multiply.reduce(a, axis=-1)
print("d:", d)

'''



















