# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import cv2
import numpy as np
import tensorflow as tf 
from matplotlib import pyplot as plt
from drgk_anomaly.networks_keras import GanomalyGAN as ggan
#from drgk_anomaly.loss import l2_loss,l1_loss,bce_loss
#slim = tf.contrib.slim
KL=tf.keras.layers
KB=tf.keras.backend
KM=tf.keras.models
KO=tf.keras.optimizers
KLOSS = tf.keras.losses
KLoss = tf.keras.losses

def l1_loss(y_true,y_pred):
    return KLoss.mean_absolute_error(y_true,y_pred)

def l2_loss(y_true,y_pred):
    return KLoss.mean_squared_error(y_true,y_pred)


def bce_loss(y_true,y_pred,label_smoothing=0.25):
    #loss= tf.losses.sigmoid_cross_entropy(logits=logits,multi_class_labels=labels,label_smoothing=label_smoothing)
    #return tf.reduce_mean(loss)
    bceloss = KLoss.binary_crossentropy(y_true,y_pred)
    return bceloss


class GanomalyGan_Model(object):
    """Train,Eval... the network.
    """
    def __init__(self,image_size=128,batch_size=64,num_outputs=3,depth=64, z_dim=None):
        # 定义输入
        #self.x = KL.Input(shape=(batch_size,image_size,image_size,num_outputs),name='x')
        # 
        if z_dim == None : 
            z_dim= (image_size / 2)**2

        self._depth = depth
        self._z_dim = z_dim
        self._image_size = image_size
        self._num_outputs = num_outputs
        self._batch_size=batch_size

        self._lr=0.0002
        self._beta1=0.5
        self._beta2=0.9999
        #
        self.networks=ggan(   
            depth = depth,
            z_dim = int(z_dim),
            image_size= image_size,  # 事实上这定义了输入、生成图像的规格！
            num_outputs= num_outputs)
        
        # 
        self.true_label = np.ones((self._batch_size,1))
        self.false_label = np.zeros((self._batch_size,1))
        self.netG,self.netG_E,self.netG_G = self.networks._NetG()
        self.netE = self.networks._NetE()
        self.netD = self.networks._NetD()
        self.netd_train_model=None
        self.adv_encoder_train_model=None
        self.adv_netg_g_train_model=None
        self.optimizer_netd= KO.Adam(lr=self._lr,beta_1=self._beta1,beta_2=self._beta2)
        self.optimizer_adv_encoder= KO.Adam(lr=self._lr,beta_1=self._beta1,beta_2=self._beta2)
        self.optimizer_adv_generator= KO.Adam(lr=self._lr,beta_1=self._beta1,beta_2=self._beta2)
        self.__Model()

    def __set_trainable(self,who):
        def set(net,trainable):
            net.trainable=trainable
            for layer in net.layers:
                layer.trainable=trainable   #set(layer,trainable)  # 这儿有个递归行为
            pass
        if who == 'netd':
            #set(self.netG,trainable=False)
            set(self.netG_E,trainable=False)
            set(self.netG_G,trainable=False)
            set(self.netE,trainable=False)
            set(self.netD,trainable=True)
        elif who == 'adv_generator':
            #set(self.netG,trainable=False)
            set(self.netG_E,trainable=False)
            set(self.netG_G,trainable=True)            
            set(self.netE,trainable=False)
            set(self.netD,trainable=False)
        elif who == 'adv_encoder':
            #set(self.netG,trainable=True)
            set(self.netG_E,trainable=True)
            set(self.netG_G,trainable=True)            
            set(self.netE,trainable=True)
            set(self.netD,trainable=False)            

        else : pass


    def __show_trainable_varibales(self,mod):
        for var in mod.trainable_variables:
            print(var)


    def __show_trainable_layes(self,net):
        for layer in net.layers:
            print(layer,layer.trainable)


    def __build_netd_train_model(self):
        pred_as_loss = lambda labels,preds : preds
        #netd (即netD在netG上)的训练模型:Ladv = |f(real) - f(fake)|_2   
        #那么事实上，没有训练到 C 部分
        #注意x
        #f=lambda x: l2_loss(y_true=x[:self._batch_size],y_pred=x[self._batch_size:])
        def f(x):
            #return l2_loss(y_true=x[self._batch_size:],y_pred=x[:self._batch_size])
            y=KB.concatenate([x[self._batch_size:],x[:self._batch_size]],axis=0)
            return l2_loss(y_true=y,y_pred=x)

        def feat_loss (y_true,y_pred):
            return f(y_pred)

        
        self.__set_trainable('netd')
        self.netd_train_model = self.netD
        self.netd_train_model.compile(optimizer=self.optimizer_netd
                                        ,loss=[feat_loss,bce_loss]
                                        #,metrics =['NetD/C_sigmoid_logits''accuracy']
                                        )

        #self.__show_trainable_varibales(self.netd_train_model)
        #self.netd_train_model.summary()
        pass


    def __build_adv_encoder_train_model(self):
        preds_as_loss = lambda y_true,y_pred : y_pred
        #anormal_model(即netE 在 netG 上)的训练模型: log(D(G(x)))  + ||G(x) - x|| --??        
        x  = KL.Input(shape=(self._image_size,self._image_size,self._num_outputs),name='x_g')
        z=self.netG_E(x)
        x_fake=self.netG_G(z)
        z_fake  = self.netE(x_fake)
        _,x_fake_class = self.netD(x_fake)

        #log(D(G(x)))
        #L_adv_bce = KL.Lambda(lambda x: bce_loss(y_true=self.true_label,y_pred=x),name='L_adv_bce')(x_fake_class)
        
        #||Ge(x)-E(G(x))||_2 编码误差，采用L2
        L_enc = KL.Lambda(lambda x: l2_loss(y_true=x[0],y_pred=x[1]),name = 'L_enc')([z,z_fake])
        
        #||G(x) - x||_1 重构误差，采用L1
        #L_con = KL.Lambda(lambda x: l1_loss(y_true=x[0],y_pred=x[1]),name = 'L_con')([x,x_fake])

        outputs=[x_fake_class,L_enc,x_fake]
        loss=[bce_loss,preds_as_loss,l1_loss]
        self.adv_encoder_train_model=KM.Model(inputs=x,outputs=outputs)

        self.__set_trainable('adv_encoder')  #训练那两个Encoder
        self.adv_encoder_train_model.compile(optimizer=self.optimizer_adv_encoder
                                ,loss=loss
                                #,loss_weights=[1,1,50]
                                #,metrics=['accuracy']
                                )
        
        #self.adv_train_model.summary()
        pass

    def __build_adv_netg_g_train_model(self):   #通过对抗，训练netg_g部分    
        z  = KL.Input(shape=(self._z_dim,),name='z')
        x_fake=self.netG_G(z)
        _,x_fake_class = self.netD(x_fake)

        self.adv_netg_g_train_model=KM.Model(inputs=z,outputs=x_fake_class)

        self.__set_trainable('adv_generator')
        self.adv_netg_g_train_model.compile(optimizer=self.optimizer_adv_generator,loss=bce_loss)
        
        #self.__show_trainable_varibales(self.adv_netg_g_train_model)
        #self.adv_train_model.summary()
        pass



    def __Model(self):
        """
        从原作者的论文来看，他的代码和论文是不相符的。
        netd 是单独训练的，
        netg是在netd钳制的情况下训练的。

        """
        self.__build_netd_train_model()
        self.__build_adv_encoder_train_model()
        self.__build_adv_netg_g_train_model()
    
        pass


    def __NetD_train_on_batch(self,x,y):
        #_,x_fake = self.netG.predict_on_batch(x)
        z      = np.random.uniform(-1.0, 1.0, size=[self._batch_size, self._z_dim])
        shadow = np.random.uniform(-1.0, 1.0, size=[2*self._batch_size, 4,4,self._z_dim])
        x_fake = self.netG_G.predict_on_batch(z)
        self.__set_trainable('netd')
        x_and_x_fake=np.concatenate([x,x_fake.copy()],axis=0)
        x_and_x_fake_labels=np.concatenate([self.true_label,self.false_label])
        #x_and_x_fake_labels=np.concatenate([y,1-y])
        #self.netD.summary()
        logs=self.netd_train_model.train_on_batch(x=x_and_x_fake,y=[shadow,x_and_x_fake_labels])
        print('NetD==> {}'.format(logs))


    def __Adv_train_on_batch(self,x,y):
        self.__set_trainable('adv_generator')
        #先训练netg的生成器
        z = np.random.uniform(-1.0, 1.0, size=[self._batch_size, self._z_dim])
        logs=self.adv_netg_g_train_model.train_on_batch(x=z,y=self.true_label) 
        print('Adv-g==> {}'.format(logs))
        #再训练netg的那两个Encoder
        self.__set_trainable('adv_encoder')
        logs=self.adv_encoder_train_model.train_on_batch(x=x,y=[self.true_label,np.random.uniform(-1.0, 1.0, size=[self._batch_size]),x])
        print('Adv-e==> {}'.format(logs))
        pass

    def train_on_batch(self,x,y):   
        self.__NetD_train_on_batch(x=x,y=y)
        self.__Adv_train_on_batch(x=x,y=y)

    
    def train_step(self,x,y):
        z = np.random.uniform(-1.0, 1.0, size=[self._batch_size, self._z_dim])
        pass


    def __flatten_batch_into_image(self,batch_x):
        b,h,w,c=batch_x.shape
        p=int(math.sqrt(b))
        img=np.zeros((p*h,p*w),dtype='float32')
        for i in range(p):
            for j in range(p):
                img[i*h:(i+1)*h,j*h:(j+1)*h]=batch_x[i*p+j,:,:,0]
        return img
         
    def show(self,x):
        #z = np.random.uniform(-1.0, 1.0, size=[self._batch_size, self._z_dim])
        z,x_fake=self.netG.predict_on_batch(x)
        #x_fake=self.netG_G.predict_on_batch(z)     
        all_in_one_x  = self.__flatten_batch_into_image(x)
        all_in_one_x_ = self.__flatten_batch_into_image(x_fake)
        plt.subplot(121)
        plt.imshow(all_in_one_x,cmap='gray')
        plt.subplot(122)
        plt.imshow(all_in_one_x_,cmap='gray')
        plt.tight_layout()        
        plt.show()
