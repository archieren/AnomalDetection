# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np
import cv2

import os
import tensorflow as tf

from tqdm import tqdm
from matplotlib import pyplot as plt
from drgk_anomaly import data as DS
from drgk_anomaly import model_keras


KM = tf.keras.models
KB = tf.keras.backend

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'  # or any {'0', '1', '2'}
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # tensorflow < 2.3 时,还必须设置此项,否者基本的卷积都无法运行，奇怪的事.

class Options(object):
    def __init__(self):
        self.image_size = 1024
        self.lr = 1e-4
        self.iteration = 1
        self.ckpt_dir = "ckpt"
        self.image_channel = 3
        self.depth = 64
        self.n_extra_layers = 0
        self.dataset_name = "catenary"  # "jueyuanzi" "huhe" "yibiao" "wafangdian" "oilstone" "catenary"


def build_train_data(patch_size=1024):
    dataset_name = 'catenary'
    root = os.path.join(os.getcwd(), 'work', 'ganomaly', dataset_name)
    source_dir = os.path.join(root, 'normal')
    patched_source_dir = os.path.join(root, 'normal_patches_{}'.format(patch_size))
    if os.path.exists(patched_source_dir):
        DS.clear_dir(patched_source_dir)
    DS.patch_the_image_with_save(source_dir=source_dir, output_dir=patched_source_dir, size=patch_size, bluring=False)
    pass


def resize_the_image(size):
    dataset_name = 'catenary'
    root = os.path.join(os.getcwd(), 'work', 'ganomaly', dataset_name)
    source_dir = os.path.join(root, 'temp')
    resized_source_dir = os.path.join(root, 'normal_resized_{}'.format(size))
    if os.path.exists(resized_source_dir):
        DS.clear_dir(resized_source_dir)
    DS.resize_the_images_with_save(source_dir=source_dir, output_dir=resized_source_dir, size=size)
    pass


if __name__ == '__main__':
    resize_the_image(size=1024)
    # build_train_data(patch_size=640)
