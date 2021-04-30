#!/usr/bin/env python
# coding: utf-8

# **CIFAR-10をダウンロード**
# ImageNetも使ってみよう!!

# In[18]:


from google.colab import drive
drive.mount('/content/drive')


# In[19]:


get_ipython().magic(u'cd "/content/drive/My Drive/Colab Notebooks/\u6625\u4f11\u307f\u8ab2\u984c"')


# In[ ]:


import numpy as np
import pickle


# In[ ]:


def unpickle(file):
  fo = open(file, 'rb')
  dict = pickle.load(fo, encoding='latin1')
  fo.close()
  return dict


# **Feature Scaling**
# 
# > Feature Scalingとは, 特徴量の取りうる値の範囲を変えること.以下の2つがある. 結構, 区別はあいまいらしく, 柔軟に対応しよう
# 
# 1.   正規化(normalization)
#   > [0,1]か[-1,1]という一定範囲に収めること.以下の式で, [0:1]に収まる.
# $$x_{norm, i}= \frac{x_{i}-x_{min}}{x_{max}-x_{min}}$$
# 
# 2.   標準化(standardization)
#   > 特徴量の平均を0, 分散を1に変換すること.
# $$x_{std}=\frac{x-m_{x}}{s_{x}}$$
# 
# 
# 
# 
# 

# In[ ]:


def clean(data):
  imgs = data.reshape(data.shape[0], 3, 32, 32) #1画像の3次元化
  grayscale_imgs = imgs.mean(1)   #グレースケール化(2次元化)
  cropped_imgs = grayscale_imgs[:, 4:28, 4:28]  #クロッピング(24×24)にする
  img_data = cropped_imgs.reshape(data.shape[0], -1)  #1次元化
  img_size = np.shape(img_data)[1]  #576(=24×24)
  means = np.mean(img_data, axis=1) #行平均(行ごとの平均)⇒ 各画像の平均値
  meansT = means.reshape(len(means), 1)
  stds = np.std(img_data, axis=1) #行ごとの標準偏差⇒ 各画像の標準偏差値
  stdsT = stds.reshape(len(stds), 1)
  adj_stds = np.maximum(stdsT, 1.0/np.sqrt(img_size)) #大きいほうを採用(?)
  normalized = (img_data - meansT) / adj_stds #各画像における正規化(標準化)
  return normalized  #(data.shape[0], 1)


# **CIFAR-10のデータ読み込み&クリーニング**
# 
# クリーニングは, 以下の3つを行っている.
# 1.   グレースケール化
# 2.   クロッピング
# 3.   標準化(正規化とここではしていますが...)
# 
# 
# 

# In[ ]:


def read_data(directory):
  #names = unpickle('./cifar-10-batches-py/batches.meta')['label_names']
  names = unpickle('{}/batches.meta'.format(directory))['label_names']
  print('names', names)

  data, labels = [], []
  for i in range(1, 6):  #6つのファイルをループさせる
    #filename = 'cifar-10-batches-py/data_batch_' + str(i)
    filename = '{}/data_batch_{}'.format(directory, i)
    batch_data = unpickle(filename)

    if len(data) > 0:
      data = np.vstack((data, batch_data['data']))    #垂直方向に
      labels = np.hstack((labels, batch_data['labels']))  #横方向に
    else:
      data = batch_data['data']
      labels = batch_data['labels']

  print(np.shape(data), np.shape(labels))

  data = clean(data)
  data = data.astype(np.float32)
  return names, data, labels


# In[25]:


# .ipynbファイルを.pyファイルに変更
#!jupyter nbconvert --to python 'CIFAR-10'.ipynb


# In[ ]:




