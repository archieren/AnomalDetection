# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.eager import context
import math
import tensorflow as tf
from tensorflow.python.keras.utils import tf_utils

KL = tf.keras.layers
KM = tf.keras.models
KB = tf.keras.backend
KU = tf.keras.utils
KR = tf.keras.regularizers

EXTRA_LAYERS_NUM = 0
# pylint: enable=unused-import
# 为了Spectral Normalization，需修改tensorflow.keras框架底层，暂时搞得有点乱了！


def power_iteration(W, u, rounds=1):
    '''
    Accroding the paper, we only need to do power iteration one time.
    '''
    _u = u
    for i in range(rounds):
        _v = KB.l2_normalize(KB.dot(_u, W))
        _u = KB.l2_normalize(KB.dot(_v, KB.transpose(W)))

    W_sn = KB.sum(KB.dot(KB.dot(_u, W), KB.transpose(_v)))
    return W_sn, _u, _v


class L2_Normalize(KL.Layer):
    def __init_(self, **kwargs):
        super(L2_Normalize, self).__init__(**kwargs)

    def call(self, inputs, training=True):
        return KB.l2_normalize(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape


"""
按SN的介绍，SN只用于Discriminator中，而且他和Batch Normalization有点互斥。
因此Discriminator的构造，我这儿有很多版本
但发现SN对其他东西还是有用的！
"""


class SN_Conv2D(KL.Conv2D):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 spectral_normalization=True,
                 name='SN_Conv2d',
                 **kwargs):
        super(SN_Conv2D, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            name=name,
            **kwargs)
        self.out_dim = filters
        self.spectral_normalization = spectral_normalization

    def build(self, input_shape):
        self.u = self.add_weight(shape=tuple([1, self.out_dim]), initializer='random_uniform', name="sn_estimate_u", trainable=False)
        super(SN_Conv2D, self).build(input_shape)
        self.built = True

    def compute_spectral_normal(self, training):
        # Spectrally Normalized Weight
        if self.spectral_normalization:
            W_mat = KB.reshape(self.kernel, [self.out_dim, -1])  # [out_channels, N]
            W_sn, u, _ = power_iteration(W_mat, self.u)

            def true_fn():
                self.u.assign(u)
                pass

            def false_fn():
                pass

            training_value = tf_utils.constant_value(training)
            if training_value is not None:
                tf_utils.smart_cond(training, true_fn, false_fn)
            return self.kernel/W_sn
        else:
            return self.kernel

    def call(self, inputs, training=True):
        # 略说明一下：keras model会自动将它的training参数传递到每层call的training参数中，如果层的call中有training参数。
        outputs = self._convolution_op(inputs, self.compute_spectral_normal(training=training))

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(self.bias, (1, self.filters, 1))
                    outputs += bias
                if self.rank == 2:
                    outputs = nn.bias_add(outputs, self.bias, data_format='NCHW')
                if self.rank == 3:
                    # As of Mar 2017, direct addition is significantly slower than
                    # bias_add when computing gradients. To use bias_add, we collapse Z
                    # and Y into a single dimension to obtain a 4D input tensor.
                    outputs_shape = outputs.shape.as_list()
                    if outputs_shape[0] is None:
                        outputs_shape[0] = -1
                    outputs_4d = array_ops.reshape(outputs,
                                                   [outputs_shape[0], outputs_shape[1],
                                                    outputs_shape[2] * outputs_shape[3],
                                                    outputs_shape[4]])
                    outputs_4d = nn.bias_add(outputs_4d, self.bias, data_format='NCHW')
                    outputs = array_ops.reshape(outputs_4d, outputs_shape)
            else:
                outputs = nn.bias_add(outputs, self.bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class SN_Conv2DTranspose(KL.Conv2DTranspose):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 output_padding=None,
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 name="SN_Conv2DTranspose",
                 spectral_normalization=True,
                 **kwargs):
        super(SN_Conv2DTranspose, self).__init__(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)
        self.out_dim = filters
        # self.u = KB.random_normal_variable([1, filters], 0, 1, dtype=self.dtype,name="sn_estimate")  # [1, out_channels]
        self.spectral_normalization = spectral_normalization

    def build(self, input_shape):
        self.u = self.add_weight(shape=tuple([1, self.out_dim]), initializer='random_uniform', name="sn_estimate_u_f", trainable=False)
        super(SN_Conv2DTranspose, self).build(input_shape)
        self.built = True

    def compute_spectral_normal(self, training):
        # Spectrally Normalized Weight
        if self.spectral_normalization:
            # Get the kernel tensor shape
            #  W_shape = self.kernel.shape.as_list()
            # Flatten the Tensor
            # For transpose conv, the kernel shape is [H,W,Out,In]
            # out_dim=W_shape[-2]
            W_mat = KB.reshape(self.kernel, [self.out_dim, -1])  # [out_c, N]
            sigma, u, _ = power_iteration(W_mat, self.u)

            def true_fn():
                self.u.assign(u)
                pass

            def false_fn():
                pass

            training_value = tf_utils.constant_value(training)
            if training_value is not False:
                tf_utils.smart_cond(training, true_fn, false_fn)
            return self.kernel / sigma
        else:
            return self.kernel
    """
    def call(self, inputs):
        tf.assign(self.kernel,self.compute_spectral_normal())
        return super(SN_Conv2DTranspose,self).call(inputs)
    """

    def call(self, inputs, training=True):  # 抄自系统的源码
        inputs_shape = array_ops.shape(inputs)
        batch_size = inputs_shape[0]
        if self.data_format == 'channels_first':
            h_axis, w_axis = 2, 3
        else:
            h_axis, w_axis = 1, 2

        height, width = inputs_shape[h_axis], inputs_shape[w_axis]
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.strides

        if self.output_padding is None:
            out_pad_h = out_pad_w = None
        else:
            out_pad_h, out_pad_w = self.output_padding

        # Infer the dynamic output shape:
        out_height = conv_utils.deconv_output_length(height,
                                                     kernel_h,
                                                     padding=self.padding,
                                                     output_padding=out_pad_h,
                                                     stride=stride_h,
                                                     dilation=self.dilation_rate[0])
        out_width = conv_utils.deconv_output_length(width,
                                                    kernel_w,
                                                    padding=self.padding,
                                                    output_padding=out_pad_w,
                                                    stride=stride_w,
                                                    dilation=self.dilation_rate[1])
        if self.data_format == 'channels_first':
            output_shape = (batch_size, self.filters, out_height, out_width)
        else:
            output_shape = (batch_size, out_height, out_width, self.filters)

        output_shape_tensor = array_ops.stack(output_shape)
        outputs = backend.conv2d_transpose(
            inputs,
            self.compute_spectral_normal(training=training),  # self.kernel,
            output_shape_tensor,
            strides=self.strides,
            padding=self.padding,
            data_format=self.data_format,
            dilation_rate=self.dilation_rate)

        if not context.executing_eagerly():
            # Infer the static output shape:
            out_shape = self.compute_output_shape(inputs.shape)
            outputs.set_shape(out_shape)

        if self.use_bias:
            outputs = nn.bias_add(
                outputs,
                self.bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


def hw_flatten(x):
    # Input shape x: [BATCH, HEIGHT, WIDTH, CHANNELS]
    # flat the feature volume across the width and height dimensions
    x_shape = tf.shape(x)
    return tf.reshape(x, [x_shape[0], -1, x_shape[-1]])  # return [BATCH, W*H, CHANNELS]


class SN_Attention(KL.Layer):
    def __init__(self, ch, spectral_normalization=True, **kwargs):
        super(SN_Attention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels
        self.spectral_normalization = spectral_normalization

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g, initializer='glorot_uniform', name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g, initializer='glorot_uniform', name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h, initializer='glorot_uniform', name='kernel_h')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,), initializer='zeros', name='bias_F')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,), initializer='zeros', name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,), initializer='zeros', name='bias_h')
        ###
        self.u_f = self.add_weight(shape=tuple([1, self.filters_f_g]), initializer='random_uniform', name="sn_estimate_u_f", trainable=False)  # [1, out_channels]
        self.u_g = self.add_weight(shape=tuple([1, self.filters_f_g]), initializer='random_uniform', name="sn_estimate_u_g", trainable=False)  # [1, out_channels]
        self.u_h = self.add_weight(shape=tuple([1, self.filters_h]), initializer='random_uniform', name="sn_estimate_u_h", trainable=False)  # [1, out_channels]

        super(SN_Attention, self).build(input_shape)  # 这是必须的。
        # Set input spec.
        # self.input_spec = InputSpec(ndim=4,axes={3: input_shape[-1]})
        self.built = True  # 这是必须的

    def compute_spectral_normal(self, s_kernel, s_u, training):
        # Spectrally Normalized Weight

        if self.spectral_normalization:
            W_shape = s_kernel.shape.as_list()
            out_dim = W_shape[-1]
            W_mat = KB.reshape(s_kernel, [out_dim, -1])  # [out_c, N]
            sigma, u, _ = power_iteration(W_mat, s_u)

            def true_fn():
                s_u.assign(u)
                pass

            def false_fn():
                pass

            training_value = tf_utils.constant_value(training)
            if training_value is not None:
                tf_utils.smart_cond(training, true_fn, false_fn)
            return s_kernel / sigma
        else:
            return s_kernel

    def call(self, x, training=True):
        def hw_flatten(x):
            return KB.reshape(x, shape=[KB.shape(x)[0], KB.shape(x)[1]*KB.shape(x)[2], KB.shape(x)[-1]])

        f = KB.conv2d(x,
                      kernel=self.compute_spectral_normal(s_kernel=self.kernel_f, s_u=self.u_f, training=training),
                      strides=(1, 1),
                      padding='same')  # [bs, h, w, c']
        f = KB.bias_add(f, self.bias_f)

        g = KB.conv2d(x,
                      kernel=self.compute_spectral_normal(s_kernel=self.kernel_g, s_u=self.u_g, training=training),
                      strides=(1, 1),
                      padding='same')  # [bs, h, w, c']
        g = KB.bias_add(g, self.bias_g)

        h = KB.conv2d(x,
                      kernel=self.compute_spectral_normal(s_kernel=self.kernel_h, s_u=self.u_h, training=training),
                      strides=(1, 1),
                      padding='same')  # [bs, h, w, c]
        h = KB.bias_add(h, self.bias_h)

        # s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]
        g_ = hw_flatten(g)  # [bs,N,c']
        f_ = hw_flatten(f)  # [bs,N,c']
        f_t = KB.permute_dimensions(f_, pattern=(0, 2, 1))  # [bs,c',N]
        s = KB.batch_dot(g_, f_t)  # [bs, N, N]
        beta = KB.softmax(s)  # attention map

        o = KB.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = KB.reshape(o, shape=KB.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class Attention(KL.Layer):
    def __init__(self, ch, **kwargs):
        super(Attention, self).__init__(**kwargs)
        self.channels = ch
        self.filters_f_g = self.channels // 8
        self.filters_h = self.channels

    def build(self, input_shape):
        kernel_shape_f_g = (1, 1) + (self.channels, self.filters_f_g)
        # print(kernel_shape_f_g)
        kernel_shape_h = (1, 1) + (self.channels, self.filters_h)

        # Create a trainable weight variable for this layer:
        self.gamma = self.add_weight(name='gamma', shape=[1], initializer='zeros', trainable=True)
        self.kernel_f = self.add_weight(shape=kernel_shape_f_g, initializer='glorot_uniform', name='kernel_f')
        self.kernel_g = self.add_weight(shape=kernel_shape_f_g, initializer='glorot_uniform', name='kernel_g')
        self.kernel_h = self.add_weight(shape=kernel_shape_h, initializer='glorot_uniform', name='kernel_h')
        self.bias_f = self.add_weight(shape=(self.filters_f_g,), initializer='zeros', name='bias_F')
        self.bias_g = self.add_weight(shape=(self.filters_f_g,), initializer='zeros', name='bias_g')
        self.bias_h = self.add_weight(shape=(self.filters_h,), initializer='zeros', name='bias_h')
        super(Attention, self).build(input_shape)  # 这是必须的。
        # Set input spec.
        # self.input_spec = InputSpec(ndim=4,axes={3: input_shape[-1]})
        self.built = True  # 这是必须的

    def call(self, x):
        def hw_flatten(x):
            return KB.reshape(x, shape=[KB.shape(x)[0], KB.shape(x)[1]*KB.shape(x)[2], KB.shape(x)[-1]])

        f = KB.conv2d(x, kernel=self.kernel_f, strides=(1, 1), padding='same')  # [bs, h, w, c']
        f = KB.bias_add(f, self.bias_f)

        g = KB.conv2d(x, kernel=self.kernel_g, strides=(1, 1), padding='same')  # [bs, h, w, c']
        g = KB.bias_add(g, self.bias_g)

        h = KB.conv2d(x, kernel=self.kernel_h, strides=(1, 1), padding='same')  # [bs, h, w, c]
        h = KB.bias_add(h, self.bias_h)

        # s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]
        g_ = hw_flatten(g)  # [bs,N,c']
        f_ = hw_flatten(f)  # [bs,N,c']
        f_t = KB.permute_dimensions(f_, pattern=(0, 2, 1))  # [bs,c',N]
        s = KB.batch_dot(g_, f_t)  # [bs, N, N]
        beta = KB.softmax(s)  # attention map

        o = KB.batch_dot(beta, hw_flatten(h))  # [bs, N, C]

        o = KB.reshape(o, shape=KB.shape(x))  # [bs, h, w, C]
        x = self.gamma * o + x

        return x

    def compute_output_shape(self, input_shape):
        return input_shape


class GANBuilder(object):
    """Implementation of DCGAN.
    F,Z for Encoder
    F, SN_F, Critic, SN_Critic for Discriminator

    """

    def __init__(self,
                 depth=64,
                 z_dim=100,      # 隐含向量的长度
                 image_size=64,  # 事实上这定义了输入、生成图像的规格！
                 num_outputs=3):
        """Constructor.

        Args:
            is_training: Whether the the network is for training or not.
            depth: Number of channels in last deconvolution layer(or first convolution layer) of
                the decoder(or encoder) network.
            final_size: The shape of the final output.
            num_outputs: Nuber of output features. For images, this is the
                number of channels.
            fused_batch_norm: If 'True', use a faster, fused implementation
                of batch normalization.
        """
        # self._is_training = is_training
        self._depth = depth
        self._z_dim = z_dim
        self._image_size = image_size
        self._num_outputs = num_outputs

    def F(self, name='F'):
        """F network.The common structure of the Encoder and Discriminator!
        Args:
            name:
        Returns:
            features:
            F->Z ==> Encoder
            F->C ==> Discriminator
        """
        height = self._image_size
        net_input = KL.Input(shape=(self._image_size, self._image_size, self._num_outputs), name='input')
        net = net_input

        num_layers = int(math.log(height, 2))-3
        current_depth = self._depth
        # 3->self._depth
        net = KL.Conv2D(current_depth, (4, 4), strides=2, padding='same', name='conv2d_{}'.format('init'), use_bias=False)(net)
        net = KL.LeakyReLU(0.2, name='leakyrelu_{}'.format('init'))(net)
        # 还是加一些额外层
        for i in range(EXTRA_LAYERS_NUM):
            net = KL.Conv2D(current_depth, (3, 3), strides=1, padding='same', name='extra_conv2d_{}'.format(i), use_bias=False)(net)
            net = KL.BatchNormalization(name='extra_batchnorm_{}'.format(i))(net)
            net = KL.LeakyReLU(0.2, name='extra_leakyrelu_{}'.format(i))(net)
        ##
        for i in range(num_layers//2):
            current_depth = current_depth*2
            net = KL.Conv2D(current_depth, (4, 4), strides=2, padding='same', name='conv2d_{}'.format(i+1), use_bias=False)(net)
            net = KL.BatchNormalization(name='batchnorm_{}'.format(i+1))(net)
            net = KL.LeakyReLU(0.2, name='leakyrelu_{}'.format(i+1))(net)
        # 中途加入一个Attention层!
        net = Attention(current_depth, name='attention')(net)
        # 中途加入一个Attention层!
        for i in range(num_layers//2, num_layers):
            current_depth = current_depth * 2
            net = KL.Conv2D(current_depth, (4, 4), strides=2, padding='same', name='conv2d_{}'.format(i+1), use_bias=False)(net)
            net = KL.BatchNormalization(name='batchnorm_{}'.format(i+1))(net)
            if (i == num_layers - 1):
                net = KL.LeakyReLU(0.2, name='features')(net)
            else:
                net = KL.LeakyReLU(0.2, name='leakyrelu_{}'.format(i+1))(net)
        #  令 isize = int(math.log(height, 2))
        # 此时: BNx4x4x(depth*2**( isize-3))  #注意，这和标准的DCGAN还是有些区别的！
        model = KM.Model(inputs=net_input, outputs=net, name=name)
        # model.summary()
        return model

    def SN_F(self, name='SN_F'):
        """F network.The common structure of the Encoder and Discriminator!
        Spectral Normalization 和Batch Normalization有冲突，故去掉Batch_Normalization!
        Args:
            name:
        Returns:
            features:
        """
        height = self._image_size
        net_input = KL.Input(shape=(self._image_size, self._image_size, self._num_outputs), name='input')
        net = net_input

        num_layers = int(math.log(height, 2))-3
        current_depth = self._depth
        # 3->self._depth
        net = SN_Conv2D(current_depth, (4, 4), strides=2, padding='same', name='sn_conv2d_{}'.format('init'), use_bias=False)(net)
        net = KL.LeakyReLU(0.2, name='leakyrelu_{}'.format('init'))(net)
        # 还是加一些额外层
        for i in range(EXTRA_LAYERS_NUM):
            net = SN_Conv2D(current_depth, (3, 3), strides=1, padding='same', name='extra_conv2d_{}'.format(i), use_bias=False)(net)
            net = KL.LeakyReLU(0.2, name='extra_leakyrelu_{}'.format(i))(net)
        ##
        for i in range(num_layers//2):
            current_depth = current_depth*2
            net = SN_Conv2D(current_depth, (4, 4), strides=2, padding='same', name='sn_conv2d_{}'.format(i+1), use_bias=False)(net)
            net = KL.LeakyReLU(0.2, name='leakyrelu_{}'.format(i+1))(net)
        # 中途加入一个Attention层!
        net = SN_Attention(current_depth, name='sn_attention')(net)
        # 中途加入一个Attention层!
        for i in range(num_layers//2, num_layers):
            current_depth = current_depth * 2
            net = SN_Conv2D(current_depth, (4, 4), strides=2, padding='same', name='sn_conv2d_{}'.format(i+1), use_bias=False)(net)
            if (i == num_layers - 1):
                net = KL.LeakyReLU(0.2, name='features')(net)
            else:
                net = KL.LeakyReLU(0.2, name='leakyrelu_{}'.format(i+1))(net)
        #  令 isize = int(math.log(height, 2))
        # 此时: BNx4x4x(depth*2**( isize-3))  #注意，这和标准的DCGAN还是有些区别的！
        model = KM.Model(inputs=net_input, outputs=net, name=name)
        # model.summary()
        return model

    def SN_C(self, name='SN_C'):
        height = self._image_size
        isize = int(math.log(height, 2))
        # 这个要严格和F的输出对上
        #  令 isize = int(math.log(height, 2))
        # 此时: BNx4x4x(2**( isize-3))  #注意，这和标准的DCGAN还是有些区别的！
        net_input = KL.Input(shape=(4, 4, self._depth*2**(isize-3)), name='input')
        net = net_input
        net = SN_Conv2D(1, (4, 4), strides=1, padding='VALID', use_bias=False, name='raw')(net)
        # 此时：BNx1x1x1
        net = KL.Reshape((1,), name='r_raw')(net)
        # 事实上，此处我总是有些犯糊涂，最后一层是没必要sigmoid激活的。
        # 即使在DCGAN的情况下，我们用的bce，他自己就加了sigmoid！！！
        # 此处没删，以志之。
        # net = KL.Activation('sigmoid',name='logits')(net)
        # else : net = KL.Activation('tanh',name='critics')(net)
        # 此时: BNx1
        # end_points['logits'] = net
        model = KM.Model(inputs=net_input, outputs=net, name=name)
        # model.summary()
        return model

    def Z(self, name='Z', is_vae=False):
        height = self._image_size
        isize = int(math.log(height, 2))
        #  令 isize = int(math.log(height, 2))
        # 此时: BNx4x4x(depth*2**( isize-3))  #注意，这和标准的DCGAN还是有些区别的！
        net_input = KL.Input(shape=(4, 4, self._depth*2**(isize-3)), name='input')
        net = net_input
        # 令 nz = self._z_dim
        if is_vae:
            z_mean = KL.Conv2D(self._z_dim, (4, 4), strides=1, padding='VALID', use_bias=False)(net)  # 此时：BNx1x1xnz
            z_mean = KL.Reshape((self._z_dim,), name='z_mean')(z_mean)                                # 此时：BNxnz
            z_log_var = KL.Conv2D(self._z_dim, (4, 4), strides=1, padding='VALID', use_bias=False)(net)  # 此时：BNx1x1xnz
            z_log_var = KL.Reshape((self._z_dim,), name='z_log_var')(z_log_var)                                            # 此时：BNxnz
            epsilon = KB.random_normal(shape=KB.shape(z_mean))                                                                 # 此时：BNxnz
            vae_z = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            model = KM.Model(inputs=net_input, outputs=vae_z, name=name)
        else:
            z = KL.Conv2D(self._z_dim, (4, 4), strides=1, padding='VALID', use_bias=False)(net)
            z = KL.Reshape((self._z_dim,))(z)
            model = KM.Model(inputs=net_input, outputs=z, name=name)
        # model.summary()
        return model

    def Encoder(self, name="Encoder"):  # F->Z
        f_net = self.F(name="F")
        z_net = self.Z(name="Z")

        net_input = KL.Input(shape=(self._image_size, self._image_size, self._num_outputs), name='input')
        net = net_input
        net = f_net(net)
        net = z_net(net)
        model = KM.Model(inputs=net_input, outputs=net, name=name)
        model.summary()
        return model

    def Decoder(self, name="Decoder"):
        """源自 Generator network for DCGAN.
        Construct generator network from inputs to the final endpoint.
        Args:
        """
        height = self._image_size
        num_layers = int(math.log(height, 2))-3
        current_depth = self._depth * 2**num_layers
        net_input = KL.Input(shape=(self._z_dim,), name='input')
        # num_layers = int(math.log(self._image_size, 2)) - 1
        net = KL.Reshape((1, 1, self._z_dim), name='reshape_input')(net_input)
        # input is Z, going into a convolution
        current_depth = self._depth * 2**num_layers
        net = KL.Conv2DTranspose(current_depth, (4, 4), strides=1, padding='valid', use_bias=False, name='conv2d_{}_{}'.format(self._z_dim, current_depth))(net)
        net = KL.BatchNormalization(name='batchnorm_{}'.format(current_depth))(net)
        net = KL.ReLU(name='relu_{}'.format(current_depth))(net)

        for i in range(num_layers//2):
            current_depth = current_depth // 2
            net = KL.Conv2DTranspose(current_depth, (4, 4), strides=2, padding='same', use_bias=False,
                                     name='conv2dtrans_{0}_{1}'.format(current_depth*2, current_depth))(net)
            net = KL.BatchNormalization(name='batchnorm_{}'.format(i))(net)
            net = KL.ReLU(name='relu_{}'.format(i))(net)

        # 中途加入一个Attention层!
        net = Attention(current_depth, name='attention')(net)
        # 中途加入一个Attention层!

        for i in range(num_layers//2, num_layers):
            current_depth = current_depth // 2
            net = KL.Conv2DTranspose(current_depth, (4, 4), strides=2, padding='same', use_bias=False,
                                     name='conv2dtrans_{0}_{1}'.format(current_depth*2, current_depth))(net)
            net = KL.BatchNormalization(name='batchnorm_{}'.format(i))(net)
            net = KL.ReLU(name='relu_{}'.format(i))(net)

        # 还是加一些额外层
        for i in range(EXTRA_LAYERS_NUM):
            net = KL.Conv2D(current_depth, (3, 3), strides=1, padding='same', name='extra_conv2d_{}'.format(i), use_bias=False)(net)
            net = KL.BatchNormalization(name='extra_batchnorm_{}'.format(i))(net)
            net = KL.ReLU(name='extra_relu_{}'.format(i))(net)

        net = KL.Conv2DTranspose(self._num_outputs, (4, 4), strides=2, padding='same', use_bias=False,
                                 name='conv2dtrans_{0}_{1}'.format(current_depth, current_depth//2))(net)
        net = KL.Activation('tanh', name='tanh_output')(net)
        # ----------------------
        model = KM.Model(inputs=net_input, outputs=net, name=name)

        # model.summary()
        return model

    def SN_Critic(self, name='Critic'):
        """
        按SNGAN的原始论文的介绍，Discriminator里Batch Normalization和Spectral Normalization似乎不能共存!
        故with_BN缺省值为False.
        """
        x = KL.Input(shape=(self._image_size, self._image_size, self._num_outputs), name='input')
        f_model = self.SN_F(name='F')
        feat = f_model(x)
        c_model = self.SN_C(name='C')
        critics = c_model(feat)
        model = KM.Model(inputs=x, outputs=[feat, critics], name=name)
        # model.summary()
        return model

    def _NetG(self, name='NetG'):  # NetG 指的是 Generator。F->Z->Dec
        """
        最终NetG里还是不用Spectral Normalization!
        """
        # netg
        x = KL.Input(shape=(self._image_size, self._image_size, self._num_outputs), name='input')
        # e
        netg_e_model = self.Encoder(name='E')
        z = netg_e_model(x)
        # g
        netg_g_model = self.Decoder(name='G')
        x_fake = netg_g_model(z)
        # nete
        # netg_nete_model = self.Encoder(name='NetE',format=None)
        # netg_nete_model = netg_e_model #纯粹对应论文中的结构 ，论文中netg_nete_model是同类的另外一个Encoder
        # z_fake=netg_nete_model(x_fake)

        netg_model = KM.Model(inputs=x, outputs=[z, x_fake], name=name)
        return netg_model, netg_e_model, netg_g_model

    def _NetD(self, name='NetD'):  # NetD 指的是 Discriminator。
        return self.SN_Critic(name=name)
