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
import generator
import discriminator

from six.moves import xrange
from glob import glob
from ops import *
from utils import *
from tensorflow.contrib import metrics

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class Primal(object):
  def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
         batch_size=64, sample_num=64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=48, version='basic',
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         train_mode='all', input_fname_pattern='*.jpg', checkpoint_dir=None,
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

    self.c_dim = c_dim

    self.d_bn1_1 = batch_norm(name='d_bn1_1')
    self.d_bn1_2 = batch_norm(name='d_bn1_2')
    self.d_bn1_3 = batch_norm(name='d_bn1_3')
    self.d_bn2_1 = batch_norm(name='d_bn2_1')
    self.d_bn2_2 = batch_norm(name='d_bn2_2')
    self.d_bn2_3 = batch_norm(name='d_bn2_3')
    self.d_bn3_1 = batch_norm(name='d_bn3_1')
    self.d_bn3_2 = batch_norm(name='d_bn3_2')
    self.d_bn3_3 = batch_norm(name='d_bn3_3')

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
    self.train_mode = train_mode
    self.generator = self.make_generator()
    self.sampler = self.make_sampler()
    self.discriminator = self.make_discriminator()

    self.build_model()

  def make_generator(self):
    func = lambda z,y: generator.generator_cifar(self, z, y)
    return func

  def make_sampler(self):
    func = lambda z,y: generator.sampler_cifar(self, z, y)
    return func

  def make_discriminator(self):
    func = lambda image, reuse: discriminator.disc_4layer(self, image, reuse)
    return func


  def build_model(self):
    self.y = tf.placeholder(tf.float32, [None, self.y_dim], name='y')
    self.v = tf.get_variable('v_variable', [1, self.df_dim*8*2*2],
              initializer=tf.random_normal_initializer(stddev=10))

    image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    self.labels_tf_format = tf.placeholder(tf.int64,
            [self.batch_size, self.y_dim], name='y_tfformat')

    #inputs_noise = tf.random_normal(tf.shape(self.inputs), mean=0, stddev=0.01)
    #self.inputs += inputs_noise
    sample_inputs = self.sample_inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.G = self.generator(self.z, self.y)
    h1r, h2r, fc1r, fc2r =\
        self.discriminator(self.inputs, reuse=False)

    self.input_sum = image_summary('input', self.inputs, max_outputs=10)
    self.G_sum = image_summary('G', self.G, max_outputs=10)
    layer_real = h2r
    layer_real = tf.reduce_mean(layer_real, axis=0)
    #h3_real = tf.reshape(h3_real, [1, -1])

    self.sampler = self.sampler(self.z, self.y)
    h1f, h2f, fc1f, fc2f =\
        self.discriminator(self.G, reuse=True)
    layer_fake = h2f
    layer_fake = tf.reduce_mean(layer_fake, axis=0)

    self.c_loss_real = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
          logits=fc2r, labels=self.y))

    self.c_loss_fake = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
          logits=fc2f, labels=self.y))

    diff = layer_real - layer_fake
    #diff = tf.Print(diff, [diff])
    self.feature_diff = tf.reshape(diff, [1, -1])
    self.gan_loss = -tf.matmul(self.feature_diff, tf.transpose(self.v))[0, 0]
    self.g_loss = -tf.matmul(tf.reshape(layer_fake, [1, -1]), tf.transpose(self.v))[0, 0]
    self.d_loss = self.gan_loss

    if self.train_mode == 'all':
      self.d_loss = self.gan_loss + self.c_loss_real
      self.g_loss += self.c_loss_fake + self.c_loss_real
    elif self.train_mode == 'supervised':
      self.d_loss = self.c_loss_real
    elif self.train_mode == 'gan_only':
      self.d_loss = self.gan_loss

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

    self.gan_loss_sum = scalar_summary("gan_loss", self.gan_loss)
    self.category_loss_real_sum = scalar_summary("category_loss_real", self.c_loss_real)
    self.category_loss_fake_sum = scalar_summary("category_loss_fake", self.c_loss_fake)
    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_clip_ops = [var.assign(tf.clip_by_value(var, -0.01, 0.01))\
        for var in t_vars if 'd_' in var.name]

    norm_v = tf.sqrt(tf.reduce_sum(tf.square(self.v)))
    projected_v = tf.minimum(1.0, tf.divide(1.0, norm_v)) * self.v
    self.project_v_ops = self.v.assign(projected_v)

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.v_vars = [var for var in t_vars if 'v_variable' in var.name]
    self.dfc_vars = [var for var in t_vars if 'fc2_' in var.name]
    self.d_all_vars = self.d_vars + self.dfc_vars + self.v_vars
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    #d_optim = tf.train.AdamOptimizer(1e-5, beta1=config.beta1, beta2=0.9) \
    #          .minimize(self.d_loss, var_list=self.d_all_vars)
    #g_optim = tf.train.AdamOptimizer(1e-5, beta1=config.beta1, beta2=0.9) \
    #          .minimize(self.g_loss, var_list=self.g_vars)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      d_optim = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
          .minimize(self.d_loss, var_list=self.d_all_vars)
      g_optim = tf.train.RMSPropOptimizer(learning_rate=5e-5)\
          .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    generator, eval_generator = cifar.load(config.batch_size,
            'data/cifar-10-batches-py')
    def inf_train_gen():
      while True:
        for images, labels in generator():
          yield [images, labels]
    gen = inf_train_gen()
    config.train_size = 50000

    self.g_sum = merge_summary([self.g_loss_sum, self.G_sum,
      self.category_loss_fake_sum])
    self.d_sum = merge_summary([
        self.d_loss_sum, self.gan_loss_sum,
        self.input_sum, self.category_loss_real_sum])
    self.writer = SummaryWriter(os.path.join("./logs",
        '{0}_{1}'.format(config.dataset, config.exp_name)), self.sess.graph)

    sample_inputs, lbls = gen.next()
    sample_inputs = np.reshape(sample_inputs, [-1, 3, 32, 32])
    sample_inputs = np.transpose(sample_inputs, (0, 2, 3, 1))
    sample_inputs = [transform(image,
      self.input_height, self.input_width,
      resize_height=self.output_height,
      resize_width=self.output_width, is_crop=False) for image in sample_inputs]
    sample_labels = np.zeros([config.batch_size, self.y_dim])
    sample_labels[np.arange(config.batch_size), lbls] = 1

    counter = 1
    start_time = time.time()
    ckp_dir = os.path.join(config.checkpoint_dir, config.exp_name)
    could_load, checkpoint_counter = self.load(ckp_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    sample_z = np.random.uniform(-1, 1, size=(10, self.z_dim))
    sample_z = np.tile(sample_z, [10, 1])

    for epoch in xrange(config.epoch):
      batch_idxs = config.train_size // config.batch_size
      for idx in xrange(0, batch_idxs):
        for _ in xrange(5):
          batch_images, lbls = gen.next()
          batch_images = np.reshape(batch_images, [-1, 3, 32, 32])
          batch_images = np.transpose(batch_images, (0, 2, 3, 1))
          batch_images = [transform(image,
            self.input_height, self.input_width,
            resize_height=self.output_height,
            resize_width=self.output_width, is_crop=False) for image in batch_images]
          batch_labels = np.zeros([config.batch_size, self.y_dim])
          batch_labels[np.arange(config.batch_size), lbls] = 1
          batch_z = np.random.uniform(-1, 1, size=(config.batch_size, self.z_dim))

          data_dict = { self.inputs: batch_images,
            self.z: batch_z, self.y:batch_labels,}

          self.sess.run([d_optim], feed_dict=data_dict)
          self.sess.run([self.project_v_ops])
          self.sess.run([self.d_clip_ops])

        if self.train_mode != 'supervised':
          batch_images, lbls = gen.next()
          batch_images = np.reshape(batch_images, [-1, 3, 32, 32])
          batch_images = np.transpose(batch_images, (0, 2, 3, 1))
          batch_images = [transform(image,
            self.input_height, self.input_width,
            resize_height=self.output_height,
            resize_width=self.output_width, is_crop=False) for image in batch_images]
          batch_labels = np.zeros([config.batch_size, self.y_dim])
          batch_labels[np.arange(config.batch_size), lbls] = 1
          batch_z = np.random.uniform(-1, 1, size=(config.batch_size, self.z_dim))

          data_dict = { self.inputs: batch_images,
              self.z: batch_z, self.y:batch_labels,}

          self.sess.run([g_optim], feed_dict=data_dict)

        errD, errG, errGAN, errC_real, errC_fake = self.sess.run([
          self.d_loss, self.g_loss, self.gan_loss,
          self.c_loss_real, self.c_loss_fake],
                feed_dict=data_dict)
        print 'epoch %d iter %d d_cost %0.7f gan_loss %0.7f g_cost %0.7f, c_real %0.4f, c_fake %0.4f'\
              % (epoch, idx, errD, errGAN, errG, errC_real, errC_fake)

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

        if np.mod(counter, 500) == 10:
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
