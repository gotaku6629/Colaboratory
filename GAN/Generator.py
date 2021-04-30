#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().magic(u'cd "/content/drive/My Drive/Colab Notebooks/slGan"')


# In[3]:


get_ipython().magic(u'pwd')


# In[ ]:


get_ipython().magic(u'tensorflow_version 1.x')
import tensorflow as tf
import numpy as np


# In[ ]:


def batch_norm(X, scale, offset, axes, is_train, device_name='/cpu:0'):
    # 予測のときにはそのまんまの値を返す
    if is_train is False:
        return X

    epsilon = 1e-5
    with tf.device(device_name):
        mean, variance = tf.nn.moments(X, axes)
        bn = tf.nn.batch_normalization(X, mean, variance, offset, scale,epsilon)

    return bn

# dataをreadするクラス 
class Generator(): 
    def __init__(self):

        #vgg16_model = VGG16(
        #    weights = 'imagenet',
        #    include_top = False,
        #    input_shape = (256, 192, 3)   
        #)
        #self.encoder_first = torchvision.models.vgg16(pretrained=True).features[:17] # 重み固定して使う部分
        #self.encoder_last = torchvision.models.vgg16(pretrained=True).features[17:-1] # 学習する部分
        #self.encoder_first = vgg16_model.input
        #for layer in model.layers[:17]: # 重み固定して使う部分
        #    layer.trainable = False
        #self.encoder_last = vgg16_model.output
        # Generator parameter

        self.gen_w0 = tf.Variable(
            tf.random_normal(
                shape=[100,4*4*256], stddev=0.02, dtype=tf.float32),
            name="gen_w0")

        self.gen_b0 = tf.Variable(
            tf.random_normal(
                shape=[4*4*256], stddev=0.02, dtype=tf.float32),
            name="gen_b0")    

        self.gen_w1 = tf.Variable(
            tf.random_normal(
                shape=[4, 4, 128, 256], stddev=0.02, dtype=tf.float32),
            name="gen_w1")

        self.gen_b1 = tf.Variable(
            tf.random_normal(
                shape=[128], stddev=0.02, dtype=tf.float32),
            name="gen_b1")

        self.gen_w2 = tf.Variable(
            tf.random_normal(
                shape=[4, 4, 64, 128], stddev=0.02, dtype=tf.float32),
            name="gen_w2")

        self.gen_b2 = tf.Variable(
            tf.random_normal(
                shape=[64], stddev=0.02, dtype=tf.float32),
            name="gen_b2")

        self.gen_w3 = tf.Variable(
            tf.random_normal(
                shape=[4, 4, 1, 64], stddev=0.02, dtype=tf.float32),
            name="gen_w3")

        self.gen_b3 = tf.Variable(
            tf.random_normal(
                shape=[1], stddev=0.02, dtype=tf.float32),
            name="gen_b3")          

        self.gen_scale_w1 = tf.Variable(
            tf.ones([128]), name="gen_scale_w1")
        self.gen_offset_w1 = tf.Variable(
            tf.zeros([128]), name="gen_offset_w1")

        self.gen_scale_w2 = tf.Variable(
            tf.ones([64]), name="gen_scale_w2")
        self.gen_offset_w2 = tf.Variable(
            tf.zeros([64]), name="gen_offset_w2")

        self.keep_prob = tf.placeholder(tf.float32)
        self.batch_size = tf.placeholder(tf.int32)

    def run(self, z, is_train):

        h0 = tf.reshape(tf.nn.relu(tf.nn.xw_plus_b(z, self.gen_w0, self.gen_b0)),[-1,4,4,256])

        gen_conv1 = tf.nn.conv2d_transpose(
                value=h0,
                filter=self.gen_w1,
                output_shape=[self.batch_size,7,7,128],
                strides=[1, 2, 2, 1],
                padding='SAME')+self.gen_b1

        h1 = tf.nn.leaky_relu(batch_norm(gen_conv1, self.gen_scale_w1, self.gen_offset_w1,
                        [0, 1, 2], is_train, device_name),alpha=0.2)

        gen_conv2 = tf.nn.conv2d_transpose(
                value=h1,
                filter=self.gen_w2,
                output_shape=[self.batch_size,14,14,64],
                strides=[1, 2, 2, 1],
                padding='SAME')+self.gen_b2

        h2 = tf.nn.leaky_relu(batch_norm(gen_conv2, self.gen_scale_w2, self.gen_offset_w2,
                                          [0, 1, 2], is_train, device_name),alpha=0.2)

        gen_conv3 = tf.nn.tanh(
            tf.nn.conv2d_transpose(
                value=h2,
                filter=self.gen_w3,
                output_shape=[self.batch_size,28,28,1],
                strides=[1, 2, 2, 1],
                padding='SAME')+self.gen_b3)

        return gen_conv3


# In[ ]:




