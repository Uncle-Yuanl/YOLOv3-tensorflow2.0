from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by: from config import cfg

cfg = __C

# YOLO options
__C.YOLO = edict()

__C.YOLO.CLASSES              = "D:/Anacoda/YOLO_v3_s/data/classes/coco.names"
__C.YOLO.ANCHORS              = "D:/Anacoda/YOLO_v3_s/data/anchors/baseline_anchors.txt"
__C.YOLO.STRIDES              = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE     = 3
# 影响：
__C.YOLO.IOU_LOSS_THRESHOLD = 0.5



