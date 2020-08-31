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


# tf.enable_eager_execution()
'''resize the images in batches'''


def batch_resize(imgs, size):
    img_out = np.empty([imgs.shape[0], size[0], size[1]])
    for i in range(imgs.shape[0]):
        img_out[i] = cv2.resize(imgs[i], size, interpolation=cv2.INTER_CUBIC)
    return img_out


class Options(object):
    def __init__(self):
        # self.image_size = 128        #128    64       32
        # self.z_size     = 256        #256    128      64
        # self.batch_size = 32        #         128
        self.image_size, self.z_size, self.batch_size = 64, 128, 196  # (128,256,16) (64,128,196)(32,64,1024) (16,64,1024)(8,32,1024)
        self.lr = 1e-4
        self.iteration = 2
        self.ckpt_dir = "ckpt"
        self.image_channel = 3
        self.depth = 64
        self.n_extra_layers = 0
        self.dataset_name = "wafangdian"  # "jueyuanzi" "huhe" "yibiao" "wafangdian" "oilstone"


def get_config(is_train):
    opts = Options()
    if is_train:
        opts.iteration = 2
    else:
        opts.batch_size = 2
        opts.result_dir = "result"
        opts.ckpt_dir = "ckpt"
    return opts


def normalize(x):
    # temp = (x-np.min(x))/(np.max(x)-np.min(x))
    temp = x/127.5 - 1
    return temp


def blur(imagefile, dst):
    image = cv2.imread(imagefile)
    image = cv2.GaussianBlur(image, (11, 11), 0)
    cv2.imwrite(dst, image)
    pass


def dot_self(x):
    x_t = tf.transpose(x, perm=[1, 0])
    return tf.matmul(x, x_t)


def show_mean(x):
    y = np.mean(x, axis=0)
    print(y.shape)


def init_keras_session():
    tf.enable_eager_execution()
    KB.clear_session()
    config = tf.ConfigProto()
    # 注意:Keras对内存的控制有问题!!!
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    # config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.Session(config=config)
    KB.set_session(sess)  # set this TensorFlow session as the default session for Keras


def build_train_data():
    init_keras_session()
    opts = get_config(is_train=True)
    dataset_name = opts.dataset_name
    root = os.path.join(os.getcwd(), 'work', 'ganomaly', dataset_name)
    source_dir = os.path.join(root, 'normal')
    # patched_source_dir = os.path.join(root,'normal_patches_{}'.format(opts.image_size))
    # if os.path.exists(patched_source_dir):
    #    DS.clear_dir(patched_source_dir)
    output_dir = os.path.join(root, 'normal_tf_records_{}'.format(opts.image_size))
    # DS.patch_the_image_with_save(source_dir=source_dir,output_dir=patched_source_dir,size=opts.image_size,bluring=False)
    if not os.path.exists(output_dir):   #
        os.makedirs(output_dir)
    # DS.produce_dataset_from_files(dataset_name,patched_source_dir,output_dir)
    DS.produce_dataset_from_patches_of_files(dataset_name, source_dir, output_dir, patch_size=opts.image_size, bluring=False)
    # DS.produce_dataset_from_resized_image(dataset_name, source_dir,output_dir,resized_size=opts.image_size,bluring = False)
    pass


def restore_model(model, checkpoint_dir):
    model.def_check_point(checkpoint_dir)
    model.restore_check_point(checkpoint_dir)
    pass


def train_Alocc():
    # init_keras_session()
    opts = get_config(is_train=True)
    dataset_name = opts.dataset_name
    root = os.path.join(os.getcwd(), 'work', 'ganomaly', dataset_name)

    gan_model = model_keras.Alocc_Model(image_size=opts.image_size,
                                        batch_size=opts.batch_size,  # 没用
                                        num_outputs=opts.image_channel,
                                        z_dim=opts.z_size)

    saved_image_dir = os.path.join(root, 'saved_images_{}'.format(opts.image_size))
    if not os.path.exists(saved_image_dir):   # model_dir 不应出现这种情况.
        os.makedirs(saved_image_dir)

    checkpoint_dir = os.path.join(root, 'checkpoints_{}'.format(opts.image_size))
    if not os.path.exists(checkpoint_dir):   # model_dir 不应出现这种情况.
        os.makedirs(checkpoint_dir)
    gan_model.def_check_point(checkpoint_dir)
    if tf.train.latest_checkpoint(checkpoint_dir) is not None:
        gan_model.restore_check_point(checkpoint_dir)

    #
    dataset_filepath = os.path.join(root, 'normal_tf_records_{}'.format(opts.image_size), '{}.tfrecord'.format(opts.dataset_name))
    print(dataset_filepath)
    dataset = tf.data.TFRecordDataset(dataset_filepath)
    dataset = dataset.map(DS.parser).shuffle(buffer_size=200000)
    # dataset = dataset.repeat(opts.iteration)
    dataset = dataset.batch(opts.batch_size)

    for epoch in range(opts.iteration):
        bar = tqdm(dataset)
        for (batch_n, batch_x) in enumerate(bar):
            # if batch_x.shape[0] % opts.batch_size != 0 :
            #    break
            gan_model.train_step(batch_x)
            netG_loss, L_adv, L_con, netD_loss = gan_model.get_loss()
            bar.set_description("Loss_G: {:<10f} L_adv {:<10f} L_con {:<10f} loss_D:{:<10f}".format(netG_loss, L_adv, L_con, netD_loss))
            bar.refresh()
            if batch_n % 4096 == 0:
                # print('=>>>{}'.format(batch_n))
                gan_model.generate_and_save_images(batch_n, batch_x, saved_image_dir)

        gan_model.save_check_point()
    pass


def test_Alocc_Model():
    init_keras_session()
    opts = get_config(is_train=True)
    dataset_name = opts.dataset_name
    root = os.path.join(os.getcwd(), 'work', 'ganomaly', dataset_name)

    gan_model = model_keras.Alocc_Model(image_size=opts.image_size, batch_size=opts.batch_size, num_outputs=opts.image_channel, z_dim=opts.z_size)
    checkpoint_dir = os.path.join(root, 'checkpoints_{}'.format(opts.image_size))
    restore_model(gan_model, checkpoint_dir)

    imagefile = os.path.join(root, 'abnormal', 'bird (1).jpg')  # 'jueyuanzi-1.jpg'  'oilstone1.jpg'
    testbatch, h, w = DS.patch_the_image_into_batch(imagefile, patch_size=opts.image_size, bluring=False)
    _, testbatch_fake, _ = gan_model.gen(testbatch)
    critics = gan_model.dis(testbatch_fake)
    scores = tf.sigmoid(critics)
    print(scores)

    # print(s)
    testbatch = gan_model.x_noise(testbatch)
    img = DS.depatch_patches(batch_x=testbatch, hg=h, wg=w)
    # img = DS.depatch_patches(batch_x=testbatch_fake,hg=h,wg=w)

    img = 255*(img+1)/2
    plt.subplot(2, 1, 1)
    plt.imshow(img.astype(np.uint8))
    # plt.show()

    # cos_map = cos_dis(z,z_fake)
    score_map = scores
    score_map = tf.reshape(score_map, shape=[1, h, w])

    score_map_s = tf.cast(255*score_map[0], tf.uint8)
    # print(score_map.shape)
    # print(score_map_s)
    plt.subplot(2, 1, 2)
    plt.imshow(score_map_s, cmap='gray')
    # plt.imshow((score_map*255).astype(np.uint8))
    plt.show()
    pass


def main(_):
    # build_train_data()
    # train_Alocc()
    # test_Alocc_Model()
    return


if __name__ == '__main__':
    train_Alocc()
