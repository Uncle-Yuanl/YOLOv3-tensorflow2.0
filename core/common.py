"""
some basic layer module
like: BN, CNN, Resnet, upsample
"""

import tensorflow as tf


class BatchNormalization(tf.keras.layers.BatchNormalization):
    """
    错误写法：
    Adding BatchNormalization with training=True to a model causes the result of one example to depend on the contents of all other examples in a minibatch
    def __int__(self, trainable = None):
        super(BatchNormalization, self).__init__()
        self.trainable = trainable
    """

    def call(self, x, training=False):
        """
        inference mode is normally controlled by the training argument that can be passed when calling a layer
        training=True: The layer will normalize its inputs using the mean and variance of the current batch of inputs
        training=False: The layer will normalize its inputs using the mean and variance of its moving statistics, learned during training
        train和test都会normalize，只是用不同的mean和variance，training=True时，mean和var都会因为当前batch而更新

        Frozen state" and "inference mode" are two separate concepts.`layer.trainable = False` is to freeze the layer, so the layer will use
        stored moving `var` and `mean` in the "inference mode", and both `gama` and `beta` will not be updated !
        """
        if not training:
            training = tf.constant(False)
        # 默认self.trainable = True
        # 同时True，才能更新参数 trainable = True就学，两种模式，= False， 不学就拉到
        training = tf.logical_and(training, self.trainable)
        # ******
        return super().call(x, training)


def convolutional(input_layer, filters_shape, downsample=False, activation=True, bn=True):
    # padding = "valid" or "same" 貌似不能直接传tuple/list, padding需要layer实现
    if downsample:
        input_layer = tf.keras.layers.ZeroPadding2D(padding=((1, 0), (1, 0)))(input_layer)
        strides = 2
        padding = "valid"
    else:
        strides = 1
        padding = "same"

    conv = tf.keras.layers.Conv2D(filters=filters_shape[-1], kernel_size=filters_shape[0], strides=strides, padding=padding,
                                  use_bias=not bn, kernel_regularizer=tf.keras.regularizers.l2(0.0005),
                                  # 初始化方法集合https://keras-cn.readthedocs.io/en/latest/other/initializations/
                                  # 使用初始化器避免手动输出shape 一个初始化器可以由字符串指定，或一个callable的函数
                                  kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                                  bias_initializer=tf.random_uniform_initializer(0.))(input_layer)

    if bn:
        # 参考的代码里这里会出问题，加上tf.keras.layers就没事
        # 自己写的反而不出问题了 玄学
        conv = BatchNormalization()(conv)
    if activation:
        conv = tf.nn.leaky_relu(conv, alpha=0.1)
    return conv


def residual_block(input_layer, filter_num1, filter_num2):
    short_cut = input_layer
    #                                              这边检查直接.shape应该是没问题的，返回数字，类别为int
    conv = convolutional(input_layer, filters_shape=(1, 1, input_layer.shape[-1], filter_num1))
    conv = convolutional(conv, filters_shape=(3, 3, filter_num1, filter_num2))
    output = short_cut + conv
    return output


def convolutional_set(input_layer, filter_conca, filter_num1, filter_num2):
    input_layer = convolutional(input_layer, filters_shape=(1, 1, filter_conca, filter_num2))
    input_layer = convolutional(input_layer, filters_shape=(3, 3, filter_num2, filter_num1))
    input_layer = convolutional(input_layer, filters_shape=(1, 1, filter_num1, filter_num2))
    input_layer = convolutional(input_layer, filters_shape=(3, 3, filter_num2, filter_num1))
    input_layer = convolutional(input_layer, filters_shape=(1, 1, filter_num1, filter_num2))
    return input_layer


def upsample(image):
    return tf.image.resize(image, size=(image.shape[1] * 2, image.shape[2] * 2), method='nearest')
