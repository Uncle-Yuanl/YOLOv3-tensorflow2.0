import tensorflow as tf
import numpy as np
import os
import shutil

from core.yolov3 import YOLOv3, decode, compute_loss
from core.config import cfg
from core.dataset import Dataset

"""
定义参数
"""
trainset  = Dataset('train')
logdir = "./data/log"
steps_per_epoch = len(trainset)
global_steps = tf.Variable(1, trainable=False, dtype=tf.int64)
warmup_steps = cfg.TRAIN.WARMUP_EPOCHS * steps_per_epoch
total_steps = cfg.TRAIN.EPOCHS * steps_per_epoch


"""
定义模型，optimizer和数据写入
"""
input_size = 416
input_layer = tf.keras.Input(shanpe=[input_size, input_size, 3])
conv_bboxes = YOLOv3(input_layer)
output_layers = []
for i, conv_bbox in enumerate(conv_bboxes):
    pred_bbox = decode(conv_bbox, i)
    output_layers.append(conv_bbox)  # compute_loss用
    output_layers.append(pred_bbox)

model = tf.keras.Model(inputs=input_layer, outputs=output_layers)
optimizer = tf.keras.optimizers.Adam()
if os.path.exists(logdir):
    shutil.rmtree(logdir)
writer = tf.summary.create_file_writer(logdir)


"""
定义train_step
"""
def train_step(image, target):
    # 需要将前向传播放在tape里
    with tf.GradientTape() as tape:
        output_layers = model(image, training=True)
        giou_loss = conf_loss = prob_loss = 0

        # optimizing process
        for i in range(3):
            conv, pred = output_layers[2*i], output_layers[2*i + 1]
            # 给这个函数定义多个同类的形参，在使用时，带一个*号的在方法中会被存储为元组
            # https://blog.csdn.net/wangjvv/article/details/79703509
            loss_items = compute_loss(pred, conv, *target[i], i)
            giou_loss += loss_items[0]
            conf_loss += loss_items[1]
            prob_loss += loss_items[2]

        total_loss = giou_loss + conf_loss + prob_loss
        gradients = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        tf.print("=> STEP %4d   lr: %.6f   giou_loss: %4.2f   conf_loss: %4.2f   "
                 "prob_loss: %4.2f   total_loss: %4.2f" % (global_steps.numpy(), optimizer.lr.numpy(),
                                                           giou_loss, conf_loss,
                                                           prob_loss, total_loss))

        # update learning rate
        global_steps.assign_add(1)
        if global_steps < warmup_steps:
            lr = global_steps / warmup_steps * cfg.TRAIN.LR_INIT
        else:
            lr = cfg.TRAIN.LR_END + 0.5 * (cfg.TRAIN.LR_INIT - cfg.TRAIN.LR_END) * (
                (1 + tf.cos((global_steps - warmup_steps) / (total_steps - warmup_steps) * np.pi))
            )
        # global_steps是tensor
        optimizer.lr.assign(lr.numpy())

        # writing summary data
        with writer.as_default():
            tf.summary.scalar("lr", optimizer.lr, step=global_steps)
            tf.summary.scalar("loss/total_loss", total_loss, step=global_steps)
            tf.summary.scalar("loss/giou_loss", giou_loss, step=global_steps)
            tf.summary.scalar("loss/conf_loss", conf_loss, step=global_steps)
            tf.summary.scalar("loss/prob_loss", prob_loss, step=global_steps)
        writer.flush()


"""
将dataset输入网络，得到输出，计算损失
自动求导，更新、保存参数
"""
for epoch in range(cfg.TRAIN.EPOCHS):
    for image_data, target in trainset:
        train_step(image_data, target)
    model.save_weights("./yolov3")































