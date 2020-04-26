import core.common as common
import tensorflow as tf


def darknet53(input_data):
    """
    build backbone according to the paper
    """
    input_data = common.convolutional(input_data, filters_shape=(3, 3, 3, 32))
    input_data = common.convolutional(input_data, filters_shape=(3, 3, 32, 64), downsample=True)

    for _ in range(1):
        input_data = common.residual_block(input_data, 32, 64)
    input_data = common.convolutional(input_data, filters_shape=(3, 3, 64, 128), downsample=True)

    for _ in range(2):
        input_data = common.residual_block(input_data, 64, 128)
    input_data = common.convolutional(input_data, filters_shape=(3, 3, 128, 256), downsample=True)

    for _ in range(8):
        input_data = common.residual_block(input_data, 128, 256)
    route1 = input_data
    input_data = common.convolutional(input_data, filters_shape=(3, 3, 256, 512), downsample=True)

    for _ in range(8):
        input_data = common.residual_block(input_data, 256, 512)
    route2 = input_data
    input_data = common.convolutional(input_data, filters_shape=(3, 3, 512, 1024), downsample=True)

    for _ in range(4):
        input_data = common.residual_block(input_data, 512, 1024)

    return route1, route2, input_data









