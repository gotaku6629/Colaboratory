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


class Discrimitor():
    def __init__(self):
        # Discrimitor parameter
        self.dis_w1 = tf.Variable(
            tf.random_normal(
                shape=[4, 4, 1, 64], stddev=0.02, dtype=tf.float32),
            name="dis_w1")

        self.dis_b1 = tf.Variable(
            tf.random_normal(
                shape=[64], stddev=0.02, dtype=tf.float32),
            name="dis_b1")

        self.dis_w2 = tf.Variable(
            tf.random_normal(
                shape=[4, 4, 64, 128], stddev=0.02, dtype=tf.float32),
            name="dis_w2")

        self.dis_b2 = tf.Variable(
            tf.random_normal(
                shape=[128], stddev=0.02, dtype=tf.float32),
            name="dis_b2")

        self.dis_w3 = tf.Variable(
            tf.random_normal(
                shape=[4, 4, 128, 256], stddev=0.02, dtype=tf.float32),
            name="dis_w3")

        self.dis_b3 = tf.Variable(
            tf.random_normal(
                shape=[256], stddev=0.02, dtype=tf.float32),
            name="dis_b3")

        self.dis_w4 = tf.Variable(
            tf.random_normal(
                shape=[4*4*256,1], stddev=0.02, dtype=tf.float32),
            name="dis_w4")

        self.dis_b4 = tf.Variable(
            tf.random_normal(
                shape=[1], stddev=0.02, dtype=tf.float32),
            name="dis_b4")

    def run(self, x, is_train):

        input_layer = tf.reshape(x, [-1, 28, 28, 1])
        dis_conv1 = tf.nn.conv2d(
                input=input_layer,
                filter=self.dis_w1,
                strides=[1, 2, 2, 1],
                padding='SAME')+self.dis_b1

        h1 = tf.nn.leaky_relu(dis_conv1,alpha=0.2)

        dis_conv2 = tf.nn.conv2d(
                input=h1,
                filter=self.dis_w2,
                strides=[1, 2, 2, 1],
                padding='SAME')+self.dis_b2

        h2 =tf.nn.leaky_relu(dis_conv2,alpha=0.2)     

        dis_conv3 = tf.nn.conv2d(
                input=h2,
                filter=self.dis_w3,
                strides=[1, 2, 2, 1],
                padding='SAME')+self.dis_b3

        h3 = tf.nn.leaky_relu(dis_conv3,alpha=0.2)

        h3_flat = tf.reshape(h3,[-1,4*4*256])
        fc = tf.nn.sigmoid(tf.nn.xw_plus_b(h3_flat,weights=self.dis_w4,biases=self.dis_b4))

        return fc


# In[ ]:




