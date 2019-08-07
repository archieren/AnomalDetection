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
from drgk_anomaly.networks_sn_keras import OC_NN as ocnn

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
        if z_dim is not None:
            z_dim = (image_size / 2)**2

        self._depth = depth
        self._z_dim = z_dim
        self._image_size = image_size
        self._num_outputs = num_outputs
        # self._batch_size=batch_size
        self.sigma = 0.1

        self._lr = 0.0002
        self._beta1 = 0.5
        self._beta2 = 0.9999

        self.r = 239.956100
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
        self.optimizer_netG = tf.train.AdamOptimizer(learning_rate=self._lr, beta1=self._beta1, beta2=self._beta2)
        # KO.Adam(lr=self._lr,beta_1=self._beta1,beta_2=self._beta2)

        self.netD = self.networks._NetD()
        self.optimizer_netD = tf.train.AdamOptimizer(learning_rate=self._lr, beta1=self._beta1, beta2=self._beta2)
        # KO.Adam(lr=self._lr,beta_1=self._beta1,beta_2=self._beta2)

        self.L_adv = None
        self.L_con = None
        self.netG_loss = None

        self.L_dis = None
        self.netD_loss = None

        self.__Model()

    def __Model(self):
        pass

    # @tf.function
    def train_step(self, x):
        dis_loss_fn, gen_loss_fn = ML.get_gan_losses_fn()
        with tf.GradientTape() as netG_tape, tf.GradientTape() as netD_tape:
            # 让x_noise的分布靠近x的分布.
            # 训练netD
            # x
            x_noise = self.x_noise(x)

            _, x_fake = self.netG(x, training=True)
            _, x_noise_fake = self.netG(x_noise, training=True)

            _, x_critics = self.netD(x, training=True)
            # _,x_fake_critics = self.netD(x_fake, training=True)
            _, x_noise_fake_critics = self.netD(x_noise_fake, training=True)

            # _,fake_loss=dis_loss_fn(x_critics,x_fake_critics)
            real_loss, noise_fake_loss = dis_loss_fn(x_critics, x_noise_fake_critics)
            self.L_dis = real_loss+noise_fake_loss
            self.netD_loss = self.L_dis
            grad_of_netD = netD_tape.gradient(self.netD_loss, self.netD.trainable_variables)
            self.optimizer_netD.apply_gradients(zip(grad_of_netD, self.netD.trainable_variables))
            # 训练netG
            # _,x_fake_critics = self.netD(x_fake, training=False)
            _, x_noise_fake_critics = self.netD(x_noise_fake, training=False)
            self.L_adv = gen_loss_fn(x_noise_fake_critics)
            self.L_con = l1_loss(y_true=x, y_pred=x_fake)
            self.netG_loss = 1*self.L_adv+50*self.L_con
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


class OC_NN_Model(object):
    def __init__(self, z_dim):
        self._lr = 0.0002
        self._beta1 = 0.5
        self._beta2 = 0.9999

        self.nu = 0.001
        self.r = 1.0

        self._z_dim = z_dim

        self.netOC_NN = ocnn(input_units=int(z_dim), hidden_units=int(z_dim/4), classes=1)._OC_NN(name='netOC_NN')
        self.optimizer_netOC_NN = tf.train.AdamOptimizer(learning_rate=self._lr, beta1=self._beta1, beta2=self._beta2)

        self.netOC_NN_loss = None

        self.checkpoint_prefix = None
        self.checkpoint = None

        pass

    def custom_ocnn_loss(self, nu):
        def custom_hinge(y_true, y_pred):
            # term1,term2已经定义为网络中的正则化项了.
            y_hat = KB.max(y_pred, axis=[1])
            term3 = 1 / nu * KB.mean(KB.maximum(0.0, self.r - y_hat))

            term4 = -1*self.r
            term = term3 + term4
            # yhat assigned to r
            # r = nuth quantile
            quantile = 100*nu
            # self.r = tf.contrib.distributions.percentile(y_hat,quantile) #版本上不支持,只好用numpy的.
            self.r = np.percentile(y_hat, quantile)
            # rval = KB.max(y_pred, axis=1,keepdims=True)
            # rval = tf.Print(rval, [tf.shape(rval)])
            return term
        return custom_hinge

    # @tf.function
    def train_step(self, x):
        with tf.GradientTape() as netOC_NN_tape:
            # 运行一次，取得数据！
            y_pred = self.netOC_NN(x, training=True)
            y_true = tf.zeros_like(y_pred)
            self.netOC_NN_loss = self.custom_ocnn_loss(self.nu)(y_true=y_true, y_pred=y_pred)
            grad_of_netOC_NN = netOC_NN_tape.gradient(self.netOC_NN_loss, self.netOC_NN.trainable_variables)
            self.optimizer_netOC_NN.apply_gradients(zip(grad_of_netOC_NN, self.netOC_NN.trainable_variables))
        pass

    def get_loss(self):
        return self.netOC_NN_loss

    def def_check_point(self, checkpoint_dir):
        self.checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        self.checkpoint = tf.train.Checkpoint(optimizer_netOC_NN=self.optimizer_netOC_NN, netOC_NN=self.netOC_NN)
        pass

    def save_check_point(self):
        self.checkpoint.save(file_prefix=self.checkpoint_prefix)
        pass

    def restore_check_point(self, checkpoint_dir):
        self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        pass

    def score(self, x):
        s = self.netOC_NN(x, training=False)
        return s


class FastAnoGan_Model(object):
    """
    AnoGan的后续改进,f-AnoGan。
    三部分网络Enc，Gen( = Decoder)，Dis
    gan= Gen + Dis ,

    Discriminator guided izi encoder training
    iziNetwork = (i)->Enc->(z) ->Gen ->(i') , (i,i')->Dis->(c,c')
    zizNetwork = (z)->Gen->(i) -> Enc->(z')

    """

    def __init__(self, image_size=128, batch_size=64, num_outputs=3, depth=64, z_dim=None, z_constraint=None):
        if z_dim is None:
            z_dim = (image_size / 2)**2

        self._depth = depth
        self._z_dim = int(z_dim)
        # 问题在这儿，f-AnoGan的思考点在于对z空间的控制，但似乎不对劲！！！
        # 在原文中说，z_noise为random_normal分布，Encoder输出为Tanh激活时，可以完成匹配，
        # 用于控制Encoder的输出和z_noise的生成！！！
        # 'Tanh' : 要求Encoder输出要加上Tanh激励
        #  'Norm': 要求Encoder输出要加上Norm处理 ，z_noise要加上Norm处理！
        #
        self._z_constraint = z_constraint  # 'Tanh', 'Norm', None
        self._image_size = image_size
        self._num_outputs = num_outputs
        self._batch_size = batch_size

        self._lr = 0.0002
        self._beta1 = 0.5
        self._beta2 = 0.9999

        self.networks = ggan(
            depth=depth,
            z_dim=int(z_dim),
            image_size=image_size,  # 事实上这定义了输入、生成图像的规格！
            num_outputs=num_outputs)

        # self.Encoder = self.networks.Encoder(name="Encoder",format=self._z_constraint)
        self.Encoder = self.networks.SN_Encoder(name="Encoder")
        self.optimizer_VAE = tf.train.AdamOptimizer(learning_rate=self._lr, beta1=self._beta1, beta2=self._beta2)
        self.L_vae_ziz = 0.0
        self.L_vae_izi = 0.0

        self.Gen = self.networks.SN_Decoder(name="Generator")
        self.optimizer_Gen = tf.train.AdamOptimizer(learning_rate=self._lr, beta1=self._beta1, beta2=self._beta2)
        self.L_gen_gan = 0.0

        self.Dis = self.networks.SN_Critic(name="Discriminator")
        self.optimizer_Dis = tf.train.AdamOptimizer(learning_rate=self._lr, beta1=self._beta1, beta2=self._beta2)
        self.L_dis_gan = 0.0

        self.checkpoint_dir = {'Gan': None,
                               'VAE': None}
        self.checkpoint_prefix = {'Gan': None,
                                  'VAE': None}
        self.checkpoint = {'Gan': None,
                           'VAE': None}

        return

    def def_check_point(self, checkpoint_root_dir):  # which = "Gan" or "Enc"
        self.checkpoint_dir['Gan'] = os.path.join(checkpoint_root_dir, 'Gan')
        if not os.path.exists(self.checkpoint_dir['Gan']):
            os.makedirs(self.checkpoint_dir['Gan'])
        self.checkpoint_prefix['Gan'] = os.path.join(self.checkpoint_dir['Gan'], "ckpt")
        self.checkpoint['Gan'] = tf.train.Checkpoint(
            optimizer_Gen=self.optimizer_Gen,
            optimizer_Dis=self.optimizer_Dis,
            Gen=self.Gen,
            Dis=self.Dis)

        self.checkpoint_dir['VAE'] = os.path.join(checkpoint_root_dir, 'VAE')
        if not os.path.exists(self.checkpoint_dir['VAE']):
            os.makedirs(self.checkpoint_dir['VAE'])
        self.checkpoint_prefix['VAE'] = os.path.join(self.checkpoint_dir['VAE'], "ckpt")
        self.checkpoint['VAE'] = tf.train.Checkpoint(
            optimizer_VAE=self.optimizer_VAE, Encoder=self.Encoder
            # ,Decoder=self.Decoder,
        )

        pass

    def save_check_point(self, which):
        self.checkpoint[which].save(file_prefix=self.checkpoint_prefix[which])
        pass

    def restore_check_point(self):
        if tf.train.latest_checkpoint(self.checkpoint_dir['Gan']):
            self.checkpoint['Gan'].restore(tf.train.latest_checkpoint(self.checkpoint_dir['Gan']))
        if tf.train.latest_checkpoint(self.checkpoint_dir['VAE']):
            self.checkpoint['VAE'].restore(tf.train.latest_checkpoint(self.checkpoint_dir['VAE']))
        pass

    def z_noise(self, batch_size):
        z = KB.random_normal(mean=-0.0, stddev=1.0, shape=(batch_size, self._z_dim))
        # z = KB.random_uniform(shape=(batch_size,self._z_dim),minval=-1.0,maxval=1.0)
        if self._z_constraint == 'Norm':
            z = KB.l2_normalize(z, axis=1)
        return z

    # @tf.function
    def train_gan_step_with_default(self, x):

        def gradient_penalty_loss(x, _lambda=10):
            def real_interpolated(x):
                # dragan方式
                eps = KB.random_uniform(shape=x.shape, minval=0.0, maxval=1.0)
                x_var = KB.var(x)
                x_std = KB.sqrt(x_var)
                noise = 0.5 * x_std*eps
                alpha = KB.random_uniform(shape=[x.shape[0], 1, 1, 1], minval=-1., maxval=1.)
                x_interpolated = KB.clip(x + alpha * noise, min_value=-1., max_value=1.)
                return x_interpolated

            x_interpolated = real_interpolated(x)
            with tf.GradientTape() as gp_tape:
                gp_tape.watch(x_interpolated)
                _, y_pred = self.Dis(x_interpolated, training=True)
                gradients = gp_tape.gradient(y_pred, x_interpolated)
                gradients = KB.reshape(gradients, shape=[gradients.shape[0], -1])  # 调成Batch*？,矢量形式
                gradients = KB.square(gradients)
                gradients = KB.sum(gradients, axis=1, keepdims=True)
                gradients_l2_norm = KB.sqrt(gradients)
                gradients_penalty = _lambda * KB.square(1 - gradients_l2_norm)
                gradients_penalty = KB.mean(gradients_penalty)
            return gradients_penalty

        dis_loss_fn, gen_loss_fn = ML.get_gan_losses_fn()  # ML.get_hinge_v2_losses_fn()

        current_batch_size = x.shape[0]
        # positive,negtive=KB.ones(shape=(current_batch_size,1)), KB.ones(shape=(current_batch_size,1))*(-1)
        with tf.GradientTape() as Dis_tape:
            # 训练Gan，
            # 更新Dis
            z = self.z_noise(current_batch_size)
            x_gen = self.Gen(z, training=True)  # 没有完全想明白此处为什么一定要training=True，否则训练不会成功。可以肯定地是这和BatchNormalization有关。
            _, x_critics = self.Dis(x, training=True)
            _, x_gen_critics = self.Dis(x_gen, training=True)
            real_loss, fake_loss = dis_loss_fn(x_critics, x_gen_critics)
            wgan_dis_loss = real_loss+fake_loss
            self.L_dis_gan = wgan_dis_loss
            dis_grad = Dis_tape.gradient(self.L_dis_gan, self.Dis.trainable_variables)
            self.optimizer_Dis.apply_gradients(zip(dis_grad, self.Dis.trainable_variables))

        with tf.GradientTape() as Gen_tape:
            # 更新Gen
            z = self.z_noise(current_batch_size)
            x_gen = self.Gen(z, training=True)
            _, x_gen_critics = self.Dis(x_gen, training=False)
            wgan_gen_loss = gen_loss_fn(x_gen_critics)
            self.L_gen_gan = wgan_gen_loss
            gen_grad = Gen_tape.gradient(self.L_gen_gan, self.Gen.trainable_variables)
            self.optimizer_Gen.apply_gradients(zip(gen_grad, self.Gen.trainable_variables))
        pass

    def train_vae_step_with_ziz(self, x):
        current_batch_size = x.shape[0]
        z = self.z_noise(current_batch_size)
        with tf.GradientTape() as vae_tape_ziz:
            # ziz-architecture
            x_z = self.Gen(z, training=False)
            z_x_z = self.Encoder(x_z, training=True)
            self.L_vae_ziz = ML.l2_loss(z, z_x_z)
            trainable_variables = self.Encoder.trainable_variables
            vae_ziz_grad = vae_tape_ziz.gradient(self.L_vae_ziz, trainable_variables)
            self.optimizer_VAE.apply_gradients(zip(vae_ziz_grad, trainable_variables))
        pass

    def get_loss(self):
        return self.L_gen_gan, self.L_dis_gan, self.L_vae_ziz, self.L_vae_izi

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

    def generate_and_save_images(self, epoch, test_input, output_dir):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        z = self.Encoder(test_input, training=False)
        predictions = self.Gen(z, training=False)

        image_of_input = self.__flatten_batch_into_image(test_input)
        image_of_pred = self.__flatten_batch_into_image(predictions)
        image = np.concatenate([image_of_input, image_of_pred], axis=1)
        image = 255*(image+1)/2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, 'image_at_epoch_{:04d}.jpg'.format(epoch)), image)

    def generate_and_save_images_gen(self, epoch, test_input, output_dir):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        # z = self.Encoder(test_input, training=False)
        z = self.z_noise(test_input.shape[0])
        predictions = self.Gen(z, training=False)

        image_of_input = self.__flatten_batch_into_image(test_input)
        image_of_pred = self.__flatten_batch_into_image(predictions)
        image = np.concatenate([image_of_input, image_of_pred], axis=1)
        image = 255*(image+1)/2
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, 'image_at_epoch_{:04d}_gen.jpg'.format(epoch)), image)

    def generate_images_from_zs(self, batch_size):
        z = self.z_noise(batch_size)
        print(KB.max(KB.abs(z)))
        predictions = self.Gen(z, training=False)
        image = self.__flatten_batch_into_image(predictions)
        image = 255*(image+1)/2
        image = image.astype(np.uint8)
        # image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        return image
