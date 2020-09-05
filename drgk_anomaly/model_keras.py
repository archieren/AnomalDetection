# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import cv2
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from drgk_anomaly.networks_sn_keras import GANBuilder as ggan
from drgk_anomaly import loss as ML
KL = tf.keras.layers
KB = tf.keras.backend
KM = tf.keras.models
KO = tf.keras.optimizers
KU = tf.keras.utils
KLOSS = tf.keras.losses
KLoss = tf.keras.losses


def l1_loss(y_true, y_pred):
    return KB.mean(KB.abs(y_pred - y_true))


def l2_loss(y_true, y_pred):
    return KB.mean(KB.square(y_pred - y_true))


def feat_loss(y_true, y_pred):  # 这个可能更符合文中的观点!
    y_true_mean = KB.mean(y_true, axis=0)
    y_pred_mean = KB.mean(y_pred, axis=0)
    return KB.mean(KB.abs(y_true_mean-y_pred_mean))


def bce_loss(y_true, y_pred):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))


class Alocc_Model(object):
    """Train,Eval... the network.
    """

    def __init__(self, image_size=128, batch_size=64, num_outputs=3, depth=64, z_dim=None):
        # 定义输入
        # self.x = KL.Input(shape=(batch_size,image_size,image_size,num_outputs),name='x')
        #

        self._depth = depth
        self._z_dim = z_dim
        self._image_size = image_size
        self._num_outputs = num_outputs
        # self._batch_size=batch_size
        self.sigma = 0.1

        self._lr = 0.0002
        self._beta1 = 0.5
        self._beta2 = 0.9999
        #
        self.networks = ggan(
            depth=depth,
            z_dim=int(z_dim),
            image_size=image_size,  # 事实上这定义了输入、生成图像的规格！
            num_outputs=num_outputs)

        #
        # self.true_label = tf.ones((self._batch_size,1))
        # self.false_label = tf.zeros((self._batch_size,1))
        self.netG, self.netG_E, self.netG_G = self.networks._NetG()
        # 最终我还是把nete放到netG中去了
        # 进一步，让netG_e,netG_nete为同一个网络，但从高层来看，还是和论文对应的。
        # self.netE = self.networks._NetE()
        self.optimizer_netG = KO.Adam(lr=self._lr, beta_1=self._beta1, beta_2=self._beta2)
        # tf.train.AdamOptimizer(learning_rate=self._lr, beta1=self._beta1, beta2=self._beta2)
        # KO.Adam(lr=self._lr, beta_1=self._beta1, beta_2=self._beta2)

        self.netD = self.networks._NetD()
        self.optimizer_netD = KO.Adam(lr=self._lr, beta_1=self._beta1, beta_2=self._beta2)
        # tf.train.AdamOptimizer(learning_rate=self._lr, beta1=self._beta1, beta2=self._beta2)
        # KO.Adam(lr=self._lr, beta_1=self._beta1, beta_2=self._beta2)

        self.L_adv = None
        self.L_con = None
        self.netG_loss = None

        self.L_dis = None
        self.netD_loss = None

    # @tf.function
    def train_step(self, x):
        dis_loss_fn, gen_loss_fn = ML.get_gan_losses_fn()
        with tf.GradientTape() as netG_tape, tf.GradientTape() as netD_tape:
            # 训练netD
            # x
            _, x_critics = self.netD(x, training=True)

            _, x_fake = self.netG(x, training=True)
            _, x_fake_critics = self.netD(x_fake, training=True)

            real_loss, fake_loss = dis_loss_fn(x_critics, x_fake_critics)
            self.L_dis = real_loss+fake_loss
            self.netD_loss = self.L_dis
            grad_of_netD = netD_tape.gradient(self.netD_loss, self.netD.trainable_variables)
            self.optimizer_netD.apply_gradients(zip(grad_of_netD, self.netD.trainable_variables))
            # 训练netG
            _, x_fake_critics = self.netD(x_fake, training=False)
            self.L_adv = gen_loss_fn(x_fake_critics)
            self.L_con = l2_loss(y_true=x, y_pred=x_fake)
            self.netG_loss = 1*self.L_adv+50*self.L_con+sum(self.netG.losses)
            grad_of_netG = netG_tape.gradient(self.netG_loss, self.netG.trainable_variables)
            self.optimizer_netG.apply_gradients(zip(grad_of_netG, self.netG.trainable_variables))
        pass

    def get_loss(self):
        return self.netG_loss, self.L_adv, self.L_con, self.netD_loss

    def def_check_point(self, checkpoint_dir):
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer_netG=self.optimizer_netG,
                                              optimizer_netD=self.optimizer_netD,
                                              netG=self.netG,
                                              netD=self.netD)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir, max_to_keep=3)
        pass

    def save_check_point(self):
        self.checkpoint_manager.save()
        pass

    def restore_check_point(self, checkpoint_dir):
        self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
        pass

    def __flatten_batch_into_image(self, batch_x):
        # batch_x is a tensor!
        b, h, w, c = batch_x.get_shape().as_list()
        p = int(math.sqrt(b))
        if c == 1:
            img = np.zeros((p*h, p*w), dtype='float32')
            for i in range(p):
                for j in range(p):
                    img[i*h:(i+1)*h, j*h:(j+1)*h] = batch_x[i*p+j, :, :, 0]
        else:
            img = np.zeros((p*h, p*w, c), dtype='float32')
            for i in range(p):
                for j in range(p):
                    img[i*h:(i+1)*h, j*h:(j+1)*h, :] = batch_x[i*p+j, :, :, :]
        return img

    def show(self, x, epoch):
        # x is a tensor!
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        z, x_fake = self.netG(x, training=False)
        all_in_one_x = self.__flatten_batch_into_image(x)
        all_in_one_x_ = self.__flatten_batch_into_image(x_fake)
        plt.subplot(1, 2, 1)
        plt.imshow(all_in_one_x, cmap='gray')
        plt.subplot(1, 2, 2)
        plt.imshow(all_in_one_x_, cmap='gray')
        plt.tight_layout()
        plt.show()

    def generate_and_save_images(self, epoch, test_input, output_dir):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        _, predictions = self.netG(test_input, training=False)

        image_of_input = self.__flatten_batch_into_image(test_input)
        image_of_pred = self.__flatten_batch_into_image(predictions)
        image = np.concatenate([image_of_input, image_of_pred], axis=1)
        image = 255*(image+1)/2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, 'image_at_epoch_{:04d}.jpg'.format(epoch)), image)

    def x_noise(self, x):
        shape = x.shape
        x_n = x+KB.random_normal(mean=-0.0, stddev=self.sigma**2, shape=shape)
        x_n = KB.clip(x_n, min_value=-1.0, max_value=1.0)
        return x_n

    def gen(self, x):
        z, x_fake = self.netG(x, training=False)
        _, x_fake_labels = self.netD(x_fake, training=False)

        return z, x_fake, x_fake_labels

    def dis(self, x):
        _, x_labels = self.netD(x, training=False)
        return x_labels
