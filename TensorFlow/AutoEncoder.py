#!/usr/bin/env python
# coding: utf-8

# In[1]:


from google.colab import drive
drive.mount('/content/drive')


# In[2]:


get_ipython().magic(u'cd "/content/drive/My Drive/Colab Notebooks/\u6625\u4f11\u307f\u8ab2\u984c"')


# In[3]:


get_ipython().magic(u'pwd')


# In[4]:


get_ipython().magic(u'tensorflow_version 1.x')
import tensorflow as tf
import numpy as np


# **Optimizerとは**
# 
# > tenosrflowのOptimizerは, 重み(w)やバイアス(b)などの変数を更新してくれる. そして, その更新の方法には, 確率的勾配降下法や最急降下法などさまざまな種類が用意されている. cf) オプティマイザの種類と特徴:
# https://qiita.com/cnloni/items/ad7dcb7521b936d9fc18
# 
# > 変数(Variable)の値更新には, tf.assignが有効であるが, Optimizerを使えば, その必要もない!! Optimizerを使わずに, 自力で勾配降下法を実装することももちろんできる. cf)カスタムトレーニング:
# https://www.tensorflow.org/tutorials/customization/custom_training?hl=ja
# 
# 
# 
# 
# 

# **バッチ学習**
# 
# > データ(サンプル)を1つずつネットワークに学習する(これを**オンライン学習**(ただし, ランダム選択に限る)ともいう)のではなく, 複数のデータを入力するやり方が, **バッチ学習**である. バッチ学習には, すべてのデータを一度に使う方法と, 分割して行う方法の2つがあり, 分割する方法を**ミニバッチ学習**と(これをバッチ学習とも)いう.
# 
# 
# > ミニバッチ学習は, 大きすぎるデータを分割することで, データ順序による影響を受けにくいし, 学習の停滞(局所解に陥ってしまうこと)が起きにくい(バッチ学習よりもデータ数が少なく, パラメータの変化に対応しやすい.).
# 
# > 10000件のデータを1000ずつのサブセットに分割させる場合, **バッチサイズ**は1000となる. また, バッチサイズのことを"ミニバッチサイズ"ともいう.
# 
# > オンライン学習も学習の停滞(局所解に陥ること)が起こりにくい一方, 学習の結果が不安定になりやすい(1つ1つのデータに対してパラメータの更新をするため).
# cf) https://to-kei.net/neural-network/sgd/
# 
# > ちなみに"**イテレーション数**"とは, データセットに含まれるデータが少なくとも1回は学習に用いられるのに必要な学習回数であり, バッチサイズによって決まる.10000件のデータを1000ずつに分けた場合, イテレーション数は, 10(=10000/1000)となる.
# 
# 
# 
# 
# 
# 

# In[ ]:


def get_batch(X, size):  #バッチ訓練
  a = np.random.choice(len(X), size, replace=False)
  return X[a]


# AutoEncoderのクラス宣言

# In[ ]:


class Autoencoder:

  #変数の初期化
  def __init__(self, input_dim, hidden_dim, epoch=250, learning_rate=0.001):
    self.epoch = epoch
    self.learning_rate = learning_rate

    x = tf.placeholder(dtype=tf.float32, shape=[None, input_dim]) #入力層のデータセット

    with tf.name_scope('encode'):
      weights = tf.Variable(tf.random_normal([input_dim, hidden_dim], dtype=tf.float32), name='weights')  #正規分布に従う乱数
      biases = tf.Variable(tf.zeros([hidden_dim]), name='biases')
      encoded = tf.nn.tanh(tf.matmul(x, weights) + biases)  #通常のy=wx+bに活性化関数を通して非線形性の担保

    with tf.name_scope('decode'):
      weights = tf.Variable(tf.random_normal([hidden_dim, input_dim], dtype=tf.float32), name='weights')  
      biases = tf.Variable(tf.zeros([input_dim]), name='biases')     #ここね! decode部で用意すべきバイアス項(b)は,input_dim
      decoded = tf.matmul(encoded,weights) + biases   #decode部は,そのままのy=wx+b

    self.x = x
    self.encoded = encoded
    self.decoded = decoded

    self.loss = tf.sqrt(tf.reduce_mean(tf.square(tf.subtract(self.x, self.decoded)))) #誤差関数
    self.train_op = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss) #オプティマイザの選択

    self.saver = tf.train.Saver()  #学習中の各パラメータを保存


  #def train(self, data):   #(データセットを訓練)
  def train(self, data, batch_size=10):  #バッチ訓練の適用(元データをbatch_size=10ずつに分ける)
    self.batch_size = batch_size
    #num_samples = len(data)
    with tf.Session() as sess:
      sess.run(tf.global_variables_initializer())  #変数の初期化
      for i in range(self.epoch):
        #for j in range(num_samples):
        for j in range(500):      #バッチ反復回数(十分大きめで！)
          batch_data = get_batch(data, self.batch_size)  #バッチデータの取得
          #l, _ = sess.run([self.loss, self.train_op], feed_dict={self.x: [data[j]]}) #誤差関数を求めて最適化する
          l, _ = sess.run([self.loss, self.train_op], feed_dict={self.x: batch_data}) #バッチ処理をするので!
        if i % 10 == 0:
          print('epoch {0}: loss = {1}'.format(i, l))  #このprintの仕方ね!
          self.saver.save(sess, './model.ckpt')  #学習したパラメータをファイルに保存


  def test(self, data):   #新しいデータセットで訓練
    with tf.Session() as sess:
      self.saver.restore(sess, './model.ckpt')  #学習したパラメータを読みこむ
      hidden, reconstructed = sess.run([self.encoded, self.decoded], feed_dict={self.x: data}) #エンコード&デコード処理

      print('input', data)      
      print('compressed', hidden)
      print('reconstructed', reconstructed)
      return reconstructed


# In[ ]:


#import Autoencoder
from sklearn import datasets

hidden_dim = 1
data = datasets.load_iris().data
input_dim = len(data[0])
ae = Autoencoder(input_dim, hidden_dim)
ae.train(data)
ae.test([[8,4,6,2]])


# In[ ]:




