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


get_ipython().system(u"jupyter nbconvert --to python 'Generator'.ipynb")
get_ipython().system(u"jupyter nbconvert --to python 'Discriminator'.ipynb")
import Generator as gen
import Discriminator as dis


# In[ ]:


class DCGAN():
    def __init__(self):
        self.G_is_train = tf.placeholder(tf.bool)
        self.D_is_train = tf.placeholder(tf.bool)
        self.input_X = tf.placeholder(tf.float32, shape=(None, 28 * 28))

        # t0は0のラベルを格納し、t1は1のラベルを格納する
        self.label_t0 = tf.placeholder(tf.float32, shape=(None, 1))
        self.label_t1 = tf.placeholder(tf.float32, shape=(None, 1))

        # Generator
        self.generator = gen.Generator(device_name=self.device_name)
        # 生成モデルに必要なノイズの入れ物
        self.gen_z = tf.placeholder(tf.float32, shape=(None, 100))
        # Discrimitor
        self.discrimitor = dis.Discrimitor(device_name=self.device_name)

         # weight decay
        gen_norm_term = tf.nn.l2_loss(self.generator.gen_w2) + tf.nn.l2_loss(self.generator.gen_w3)
        gen_lambda_ = 0.001

        dis_norm_term = tf.nn.l2_loss(self.discrimitor.dis_w2) + tf.nn.l2_loss(self.discrimitor.dis_w3)
        dis_lambda_ = 0.001

        # 訓練データの識別予測
        input_X = self.discrimitor.run(
                  self.input_X,
                  is_train=self.D_is_train,
                  device_name=self.device_name)
        
        # 生成されたデータの識別予測
        generated_X = self.discrimitor.run(
            self.generator.run(
                z=self.gen_z,
                is_train=self.G_is_train,
                device_name=self.device_name),
            is_train=self.D_is_train,
            device_name=self.device_name)
        
        self.dis_entropy_X = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.label_t1, logits=input_X)
        self.dis_entropy_G = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.label_t0, logits=generated_X)

        self.dis_loss = tf.reduce_mean(
            self.dis_entropy_X + self.dis_entropy_G
        ) + dis_norm_term * dis_lambda_

        self.gen_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.label_t1, logits=generated_X)
        self.gen_loss = tf.reduce_mean(
            self.gen_entropy) #+ gen_norm_term * gen_lambda_

        # 最適化する際にDならDのみのパラメータを、GならGのみのパラメータを更新するようにしたいのでモデル別の変数を取得する
        dis_vars = [
            x for x in tf.trainable_variables() if "dis_" in x.name
        ]
        gen_vars = [
            x for x in tf.trainable_variables() if "gen_" in x.name
        ]

        # 識別モデルDの最適化
        self.opt_d = tf.train.AdamOptimizer(0.0002,beta1=0.1).minimize(
                self.dis_loss, var_list=[dis_vars])
        # 生成モデルGの最適化
        self.opt_g = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(
                self.gen_loss, var_list=[gen_vars])


# In[ ]:


def train(self,
          X_train=None,
          batch_size=100,
          epoch_num=1000,
          imgpath='./mnist_DCGAN_images/',
          ckptpath='./mnist_DCGAN_checkpoints/',
          log_file='mnist_DCGAN_loss_log.csv',
          init=False):

    if X_train is None:
        raise TypeError("X_train is None")

    # 訓練途中で生成データを作成して保存したいのでその保存先の作成
    p = Path(imgpath)
    if not (p.is_dir()):
        p.mkdir()

    # モデルパラメータのチェックポイントの保存先
    ckpt_p = Path(ckptpath)
    if not (ckpt_p.is_dir()):
        ckpt_p.mkdir()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    saver = tf.train.Saver()

    sess = tf.Session()

    if (init):
        sess.run(tf.global_variables_initializer())
        print('Initialize')

    ckpt = tf.train.get_checkpoint_state(str(ckpt_p.absolute()))
    if ckpt:
        # checkpointファイルから最後に保存したモデルへのパスを取得する
        last_model = ckpt.model_checkpoint_path
        print("load {0}".format(last_model))
        # 学習済みモデルを読み込む
        saver.restore(sess, last_model)

    step = len(X_train) // batch_size
    #step = mnist.train.num_examples // batch_size

    # 正解ラベルのミニバッチ
    t1_batch = np.ones((batch_size, 1), dtype=np.float32)
    t0_batch = np.zeros((batch_size, 1), dtype=np.float32)

    for epoch in range(epoch_num):

        perm = np.random.permutation(len(X_train))
        # １エポックごとにかかる時間の計測
        start = time.time()
        for k in range(step):
            #X_batch = mnist.train.next_batch(batch_size)[0] /255.
            X_batch = X_train[perm][k * batch_size:(k + 1) * batch_size]

            # Train Discrimitor
            # ノイズ事前分布からノイズをミニバッチ分取得
            noise_z = np.random.uniform(
                -1, 1, size=[batch_size,100]).astype(np.float32)

            sess.run(
                self.opt_d,
                feed_dict={
                    self.input_X: X_batch,
                    self.D_is_train: True,
                    self.G_is_train: False,
                    self.gen_z: noise_z,
                    self.generator.keep_prob: 1.0,
                    self.generator.batch_size: batch_size,
                    self.label_t1: t1_batch,
                    self.label_t0: t0_batch
                })

            if k % 1 == 0:
                # Train Generator
                # ノイズ事前分布からノイズをミニバッチ分取得
                noise_z = np.random.uniform(
                    -1, 1, size=[batch_size,100]).astype(np.float32)
                sess.run(
                    self.opt_g,
                    feed_dict={
                        self.gen_z: noise_z,
                        self.D_is_train: False,
                        self.G_is_train: True,
                        self.generator.keep_prob: 0.5,
                        self.generator.batch_size: batch_size,
                        self.label_t1: t1_batch
                    })

        # 1epoch終了時の損失を表示
        noise_z = np.random.uniform(
            -1, 1, size=[batch_size,100]).astype(np.float32)
        train_dis_loss = sess.run(
            self.dis_loss,
            feed_dict={
                self.input_X: X_batch,
                self.D_is_train: False,
                self.G_is_train: False,
                self.gen_z: noise_z,
                self.generator.keep_prob: 1.0,
                self.generator.batch_size: batch_size,
                self.label_t1: t1_batch,
                self.label_t0: t0_batch
            })

        train_gen_loss = sess.run(
            self.gen_loss,
            feed_dict={
                self.gen_z: noise_z,
                self.D_is_train: False,
                self.G_is_train: False,
                self.generator.keep_prob: 1.0,
                self.generator.batch_size: batch_size,
                self.label_t1: t1_batch
            })
        print(
            "[Train] epoch: %d, dis loss: %f , gen loss : %f  Time : %f" %
            (epoch, train_dis_loss, train_gen_loss, time.time() - start))

        f = open(log_file, 'a')
        log_writer = csv.writer(f, lineterminator='\n')
        loss_list = []
        loss_list.append(epoch)
        loss_list.append(train_dis_loss)
        loss_list.append(train_gen_loss)
        # 損失の値を書き込む
        log_writer.writerow(loss_list)
        f.close()

        saver.save(sess, str(ckpt_p.absolute()) + '/DCGAN-mnist')

        # 10epoch終了毎に生成モデルから1枚の画像を生成する
        if epoch % 2 == 0:
            noise_z = np.random.uniform(
                -1,1, size=[5,100]).astype(np.float32)

            z_const = tf.constant(noise_z, dtype=tf.float32)
            gen_imgs = ((sess.run(
                self.generator.run(z_const, is_train=False),
                feed_dict={self.generator.keep_prob: 1.0,self.generator.batch_size: 5})* 0.5)+0.5)*255.
            for i in range(0,5):
                Image.fromarray(gen_imgs[i].reshape(
                    28, 28)).convert('RGB').save(
                        str(p.absolute()) +
                        '/generate_img_epoch{0}_{1}.jpg'.format(epoch,i))


# In[ ]:




