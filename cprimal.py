from __future__ import division
import cifar
import generator
import math
import numpy as np
import os
import pdb
import socket
import tensorflow as tf
import time

from six.moves import xrange
from glob import glob
from ops import *
from utils import *
from tensorflow.contrib import metrics
from generator import generator_mnist, generator_cifar, sampler_cifar

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class CPrimal(object):
  def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64, version='basic',
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         with_label_loss=True, input_fname_pattern='*.jpg', checkpoint_dir=None,
         exp_name='basic', sample_dir=None):
    """

    Args:
      sess: TensorFlow session
      batch_size: The size of batch. Should be specified before training.
      y_dim: (optional) Dimension of dim for y. [None]
      z_dim: (optional) Dimension of dim for Z. [100]
      gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
      df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
      gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
      dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
      c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
    """
    self.sess = sess
    self.is_crop = is_crop
    self.is_grayscale = (c_dim == 1)

    self.batch_size = batch_size
    self.sample_num = sample_num

    self.input_height = input_height
    self.input_width = input_width
    self.output_height = output_height
    self.output_width = output_width

    self.y_dim = y_dim
    self.z_dim = z_dim

    self.gf_dim = gf_dim
    self.df_dim = df_dim

    self.gfc_dim = gfc_dim
    self.dfc_dim = dfc_dim

    self.c_dim = c_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    self.d_bn3 = batch_norm(name='d_bn3')

    self.g_bn0 = batch_norm(name='g_bn0')
    self.g_bn1 = batch_norm(name='g_bn1')
    self.g_bn2 = batch_norm(name='g_bn2')

    self.g_bn3 = batch_norm(name='g_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.exp_name = exp_name
    self.version = version
    self.LAMBDA = 10
    self.with_label_loss = with_label_loss
    self.generator = self.make_generator()
    self.sampler = self.make_sampler()

    self.build_model()

  def make_generator(self):
    func = lambda z,y: generator_cifar(self, z, y)
    return func

  def make_sampler(self):
    func = lambda z,y: sampler_cifar(self, z, y)
    return func

  def build_model(self):
    self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')
    self.v = tf.get_variable('v_variable', [10, self.df_dim*4*4*4],
              initializer=tf.random_normal_initializer(stddev=10))

    image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    self.labels_tf_format = tf.placeholder(tf.int64,
            [self.batch_size, self.y_dim], name='y_tfformat')

    inputs_noise = tf.random_normal(tf.shape(self.inputs), mean=0, stddev=1)
    self.inputs += inputs_noise
    sample_inputs = self.sample_inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.G = self.generator(self.z, self.y)
    h1r, h2r, fc1r, fc2r =\
        self.discriminator(self.inputs, self.y, reuse=False)
    bs_per_tag = int(self.batch_size/10)
    layer_real = h2r
    layer_real = tf.reshape(layer_real, [10, bs_per_tag, 4, 4, self.df_dim*4])
    layer_real = tf.reduce_mean(layer_real, axis=1)
    layer_real = tf.reshape(layer_real, [10, -1])
    #h3_real = tf.reshape(h3_real, [1, -1])

    self.sampler = self.sampler(self.z, self.y)
    h1f, h2f, fc1f, fc2f =\
        self.discriminator(self.G, self.y, reuse=True)
    layer_fake = h2f
    layer_fake = tf.reshape(layer_fake, [10, bs_per_tag, 4, 4, self.df_dim*4])
    layer_fake = tf.reduce_mean(layer_fake, axis=1)
    layer_fake = tf.reshape(layer_fake, [10, -1])

    self.c_loss_real = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
          logits=fc2r, labels=self.y))

    self.c_loss_fake = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
          logits=fc2f, labels=self.y))

    diff = layer_real - layer_fake
    # 10x1
    self.error_all_tags = tf.reduce_sum(tf.multiply(self.v, diff), axis=1)
    error = tf.reduce_mean(self.error_all_tags) # 1x1
    self.d_loss = -error

    error_fake = tf.reduce_sum(tf.multiply(self.v, -layer_fake), axis=1)
    self.g_loss = tf.reduce_mean(error_fake)

    self.all_d_loss_sum = []
    for i in range(10):
        self.all_d_loss = scalar_summary("g_loss_%d" % i, self.error_all_tags[i])

    if self.with_label_loss:
      self.d_loss += self.c_loss_real + self.c_loss_fake
      self.g_loss += self.c_loss_fake

    # evaluations
    cat_logits = tf.nn.sigmoid(fc2r)
    prec_at_t, prec_t_update_op =\
            metrics.streaming_precision_at_thresholds(
                    cat_logits, self.y, [0.2])
    prec_at_k, prec_k_update_op =\
            metrics.streaming_sparse_precision_at_k(
                    cat_logits, self.labels_tf_format, 1)
    recall_at_k, recall_k_update_op =\
            metrics.streaming_sparse_recall_at_k(
                    cat_logits, self.labels_tf_format, 1)

    prec_t_sum = scalar_summary("precision@0.2", prec_at_t[0])
    prec_k_sum = scalar_summary("precision@1", prec_at_k)
    recall_k_sum = scalar_summary("recall@1", recall_at_k)
    self.metrics_sum = merge_summary([prec_t_sum, prec_k_sum, recall_k_sum])
    self.update_ops = [prec_t_update_op, prec_k_update_op, recall_k_update_op]

    self.category_loss_real_sum = scalar_summary("category_loss_real", self.c_loss_real)
    self.category_loss_fake_sum = scalar_summary("category_loss_fake", self.c_loss_fake)
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_clip_ops = [var.assign(tf.clip_by_value(var, -0.01, 0.01))\
        for var in t_vars if 'd_' in var.name]

    norm_v = tf.sqrt(tf.reduce_sum(tf.square(self.v), axis=1))
    norm_v = tf.expand_dims(norm_v, 1)
    norm_v = tf.tile(norm_v, [1, self.df_dim*4*4*4])
    projected_v = tf.minimum(1.0, tf.divide(1.0, norm_v)) * self.v
    #norm_v = tf.Print(norm_v, [norm_v])
    #projected_v = tf.Print(projected_v, [projected_v])
    self.project_v_ops = self.v.assign(projected_v)

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.v_vars = [var for var in t_vars if 'v_variable' in var.name]
    self.dfc_vars = [var for var in t_vars if 'fc2_' in var.name]
    self.d_all_vars = self.d_vars + self.dfc_vars + self.v_vars
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def get_batch_data(self, gen):
    inputs, lbls = gen.next()
    inputs = np.reshape(inputs, [self.batch_size, -1])
    inputs = np.reshape(inputs, [-1, 3, 32, 32])
    inputs = np.transpose(inputs, (0, 2, 3, 1))
    inputs = [transform(image, self.input_height, self.input_width,
      resize_height=self.output_height, resize_width=self.output_width, is_crop=False)
      for image in inputs]
    labels = np.zeros([self.batch_size, self.y_dim])
    labels[np.arange(self.batch_size), lbls] = 1
    return inputs, labels

  def train(self, config):
    #d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=0.9) \
    #          .minimize(self.d_loss, var_list=self.d_vars)
    #g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1, beta2=0.9) \
    #          .minimize(self.g_loss, var_list=self.g_vars)

    d_optim = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
        .minimize(self.d_loss, var_list=self.d_all_vars)
    g_optim = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
        .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    generator, eval_generator = cifar.load(config.batch_size,
            'data/cifar-10-batches-py',
            with_cond=True)
    def inf_train_gen():
      while True:
        for images, labels in generator():
          yield [images, labels]
    gen = inf_train_gen()
    config.train_size = 50000

    self.g_sum = merge_summary([self.g_loss_sum,
      self.category_loss_fake_sum])
    self.d_sum = merge_summary([self.d_loss_sum,
        self.category_loss_real_sum] + self.all_d_loss_sum)
    self.writer = SummaryWriter(os.path.join("./logs",
        '{0}_{1}'.format(config.dataset, config.exp_name)), self.sess.graph)

    sample_inputs, sample_labels = self.get_batch_data(gen)
    counter = 1
    start_time = time.time()
    ckp_dir = os.path.join(config.checkpoint_dir, config.exp_name)
    could_load, checkpoint_counter = self.load(ckp_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    sample_z = np.random.normal(0, 1, size=(10, self.z_dim))
    sample_z = np.tile(sample_z, [10, 1])

    for epoch in xrange(config.epoch):
      batch_idxs = config.train_size // config.batch_size
      for idx in xrange(0, batch_idxs):
        for _ in xrange(5):
          batch_images, batch_labels = self.get_batch_data(gen)
          batch_z = np.random.normal(0, 1, size=(config.batch_size, self.z_dim))

          data_dict = { self.inputs: batch_images,
            self.z: batch_z, self.y:batch_labels,}

          self.sess.run([d_optim], feed_dict=data_dict)
          self.sess.run([self.project_v_ops])
          self.sess.run([self.d_clip_ops])

        batch_images, batch_labels = self.get_batch_data(gen)
        batch_z = np.random.normal(0, 1, size=(config.batch_size, self.z_dim))

        data_dict = { self.inputs: batch_images,
            self.z: batch_z, self.y:batch_labels,}

        self.sess.run([g_optim], feed_dict=data_dict)

        errD, errG, err_all, errC_real, errC_fake = self.sess.run([
          self.d_loss, self.g_loss, self.error_all_tags,
          self.c_loss_real, self.c_loss_fake],
                feed_dict=data_dict)
        mean = np.mean(err_all)
        minv = np.min(err_all)
        maxv = np.max(err_all)
        print 'epoch %d iter %d d_cost %0.7f (%0.7f %0.7f %0.7f) g_cost %0.7f, c_real %0.4f, c_fake %0.4f'\
              % (epoch, idx, errD, minv, mean, maxv, errG, errC_real, errC_fake)

        counter += 1

        if np.mod(counter, 100) == 1:
          sample_labels = np.zeros((self.sample_num, self.y_dim)).astype(np.float32)
          sample_labels[np.arange(self.sample_num), np.divide(np.arange(self.sample_num), 10)] = 1
          samples = self.sess.run(
            self.sampler,
            feed_dict={
                self.z: sample_z,
                self.y:sample_labels,
            }
          )
          manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
          manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
          save_images(samples, [manifold_h, manifold_w],
              './{}/{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, config.exp_name,
                    epoch, idx))

        if np.mod(counter, 50) == 49:
          summary_str = self.sess.run(self.d_sum, feed_dict=data_dict)
          self.writer.add_summary(summary_str, counter)

          summary_str = self.sess.run(self.g_sum, feed_dict=data_dict)
          self.writer.add_summary(summary_str, counter)

        if np.mod(counter, 200) == 2:
          ckp_dir = os.path.join(config.checkpoint_dir, config.exp_name)
          self.save(ckp_dir, counter)

        if np.mod(counter, 10) == 2: # eval
            self.eval(eval_generator, counter)

  def eval(self, generator, counter):
    done = False
    names = ["precision@0.2", "precision@1", "recall@1"]
    generator = generator()
    tf.local_variables_initializer().run()
    for batch_imgs, lbls in generator:
      batch_imgs = np.reshape(batch_imgs, [-1, 3, 32, 32])
      batch_imgs = np.transpose(batch_imgs, (0, 2, 3, 1))
      batch_imgs = [transform(image,
        self.input_height, self.input_width,
        resize_height=self.output_height,
        resize_width=self.output_width, is_crop=False) for image in batch_imgs]
      embed = np.zeros([self.batch_size, 10], dtype=np.int64)
      embed[np.arange(self.batch_size), lbls] = 1
      embed2 = -1*np.ones([self.batch_size, 10], dtype=np.int64)
      embed2[:, 0] = lbls

      values = self.sess.run(self.update_ops,
              feed_dict={
                  self.inputs: batch_imgs, self.y: embed,
                  self.labels_tf_format: embed2})
      values[0] = values[0][0]
      values_str = map(lambda v: "%0.4f" % v, values)
      metrics_str = [val for pair in zip(names, values_str) for val in pair]
      metrics_str = " ".join(metrics_str)

      print 'Eval checkpoint %d %s' % (counter, metrics_str)
    summary_str = self.sess.run(self.metrics_sum)
    self.writer.add_summary(summary_str, counter)


  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = conv2d(h0, self.df_dim*2, name='d_h1_conv')
      h1 = self.d_bn1(h1)
      h1 = lrelu(h1)

      h2 = conv2d(h1, self.df_dim*4, name='d_h2_conv')
      h2 = self.d_bn2(h2)
      h2 = lrelu(h2)

      fc1 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_fc1_lin')
      fc2 = linear(tf.reshape(h2, [self.batch_size, -1]), self.y_dim, 'fc2_lin')

      return h1, h2, fc1, fc2

  @property
  def model_dir(self):
    return "{}_{}_{}_{}".format(
        self.dataset_name, self.batch_size,
        self.output_height, self.output_width)

  def save(self, checkpoint_dir, step):
    model_name = "DCGAN.model"
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(self.sess,
            os.path.join(checkpoint_dir, model_name),
            global_step=step)

  def load(self, checkpoint_dir):
    import re
    print(" [*] Reading checkpoints...")
    checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
      counter = int(next(re.finditer("(\d+)(?!.*\d)",ckpt_name)).group(0))
      print(" [*] Success to read {}".format(ckpt_name))
      return True, counter
    else:
      print(" [*] Failed to find a checkpoint")
      return False, 0
