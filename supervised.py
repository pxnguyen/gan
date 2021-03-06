from __future__ import division
import math
import numpy as np
import os
import pdb
import random
import socket
import tensorflow as tf
import time

from six.moves import xrange
from glob import glob
from ops import *
from utils import *
from tensorflow.contrib import metrics

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class FullySupervised(object):
  def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64, version='fm1.0',
         gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
         input_fname_pattern='*.jpg', checkpoint_dir=None, exp_name='basic', sample_dir=None):
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

    self.df_dim = df_dim

    self.dfc_dim = dfc_dim

    self.c_dim = c_dim

    # batch normalization : deals with poor initialization helps gradient flow
    self.d_bn1 = batch_norm(name='d_bn1')
    self.d_bn2 = batch_norm(name='d_bn2')

    self.d_bn3 = batch_norm(name='d_bn3')

    self.dataset_name = dataset_name
    self.input_fname_pattern = input_fname_pattern
    self.checkpoint_dir = checkpoint_dir
    self.exp_name = exp_name
    self.version = version

    self.build_model()

  def build_model(self):
    self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

    if self.is_crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')

    inputs = self.inputs

    self.real_cat_logits = self.discriminator(inputs, self.y, reuse=False)
    self.inputs_sum = image_summary("inputs", inputs)

    def sigmoid_cross_entropy_with_logits(x, y):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

    self.category_loss_real = tf.reduce_mean(
        sigmoid_cross_entropy_with_logits(
          self.real_cat_logits,
          self.y))
    self.category_loss_real = tf.Print(self.category_loss_real,
            [self.y, self.real_cat_logits, self.category_loss_real], summarize=10)

    self.category_loss_real_sum = scalar_summary("category_loss_real", self.category_loss_real)

    self.d_loss = self.category_loss_real
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()
    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    print [var.name for var in self.d_vars]
    self.saver = tf.train.Saver()

  def train(self, config):
    if config.dataset == 'nuswide':
      label_dict = load_label_dict(config.data_dir, 'train')
      data_X = glob(os.path.join(config.data_dir, "train", self.input_fname_pattern))
      data_y = load_labels(label_dict, data_X, 14)

    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    try:
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.d_sum = merge_summary(
        [self.d_loss_sum, self.inputs_sum, self.category_loss_real_sum])
    self.writer = SummaryWriter(os.path.join("./logs",
        '{0}_{1}'.format(config.dataset, config.exp_name)), self.sess.graph)

    counter = 1
    start_time = time.time()
    ckp_dir = os.path.join(config.checkpoint_dir, config.exp_name)
    could_load, checkpoint_counter = self.load(ckp_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    population = range(len(data_X))

    for epoch in xrange(config.epoch):
      batch_idxs = min(len(data_X), config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        indeces = random.sample(population, config.batch_size)
        batch_files = [data_X[i] for i in indeces]
        #batch_files = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
        batch = [get_image(
            batch_file,
            input_height=self.input_height,
            input_width=self.input_width,
            resize_height=self.output_height,
            resize_width=self.output_width,
            is_crop=self.is_crop,
            is_grayscale=False) for batch_file in batch_files]

        batch_images = np.array(batch).astype(np.float32)
        batch_labels = data_y[indeces]
        #batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]

        data_dict = {
          self.inputs: batch_images,
          self.y:batch_labels,
        }
        # Update D network
        self.sess.run([d_optim], feed_dict=data_dict)

        if np.mod(counter, 50) == 10:
          summary_str = self.sess.run(self.d_sum, feed_dict=data_dict)
          self.writer.add_summary(summary_str, counter)

        errD_real = self.category_loss_real.eval({
            self.inputs: batch_images,
            self.y:batch_labels
        })

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_real))

        if np.mod(counter, 200) == 2:
          ckp_dir = os.path.join(config.checkpoint_dir, config.exp_name)
          self.save(ckp_dir, counter)

  def eval(self, config):
    """Eval DCGAN"""
    dataset_dir = os.path.join(config.data_dir, config.dataset)
    label_dict = load_label_dict(dataset_dir, 'eval')
    data_X = glob(os.path.join(dataset_dir, "eval", self.input_fname_pattern))
    data_y = load_labels(label_dict, data_X, 14)
    data_y_tf_format = load_labels_tf_format(label_dict, data_X, 14)

    self.writer = SummaryWriter(os.path.join("./logs",
        '{0}_{1}_eval'.format(config.dataset, config.exp_name)), self.sess.graph)

    batch_idxs = min(len(data_X), config.train_size) // config.batch_size
    cat_scores = self.discriminator(self.inputs, reuse=True)
    cat_logits = tf.nn.sigmoid(cat_scores)
    self.labels_tf_format = tf.placeholder(tf.int64, [self.batch_size, self.y_dim], name='y_tfformat')

    prec_at_t, prec_t_update_op = metrics.streaming_precision_at_thresholds(cat_logits, self.y, [0.2])
    prec_at_k, prec_k_update_op = metrics.streaming_sparse_precision_at_k(cat_logits,
            self.labels_tf_format, 1)
    recall_at_k, recall_k_update_op = metrics.streaming_sparse_recall_at_k(cat_logits,
            self.labels_tf_format, 1)

    names = ["precision@0.2", "precision@1", "recall@1"]
    update_ops = [prec_t_update_op, prec_k_update_op, recall_k_update_op]

    prec_t_sum = scalar_summary("precision@0.2", prec_at_t[0])
    prec_k_sum = scalar_summary("precision@1", prec_at_k)
    recall_k_sum = scalar_summary("recall@1", recall_at_k)

    metrics_sum = merge_summary([prec_t_sum, prec_k_sum, recall_k_sum])

    done = False
    last_counter = -1
    while not done:
      tf.global_variables_initializer().run()
      tf.local_variables_initializer().run()
      ckp_dir = os.path.join(config.checkpoint_dir, config.exp_name)
      could_load, checkpoint_counter = self.load(ckp_dir)

      if could_load:
        counter = checkpoint_counter
        if counter == last_counter:
          print "Finding the same number, sleeping..."
          time.sleep(600)
          continue
        else:
          last_counter = counter
          print(" [*] Load SUCCESS")
      else:
        raise Exception(" [!] No model found...")

      population = range(len(data_X))

      for idx in xrange(0, batch_idxs):
        indeces = random.sample(population, config.batch_size)
        batch_files = [data_X[i] for i in indeces]
        #batch_files = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
        batch = [
            get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      is_crop=self.is_crop,
                      is_grayscale=False) for batch_file in batch_files]

        batch_images = np.array(batch).astype(np.float32)
        batch_labels = data_y[indeces]
        batch_lbls_tf = data_y_tf_format[indeces]

        values = self.sess.run(update_ops, feed_dict={
          self.inputs: batch_images, self.y: batch_labels,
          self.labels_tf_format: batch_lbls_tf})
        values[0] = values[0][0]

        values_str = map(lambda v: "%0.4f" % v, values)
        metrics_str = [val for pair in zip(names, values_str) for val in pair]
        metrics_str = " ".join(metrics_str)

        print 'Eval checkpoint %d [%4d/%4d] %s' % (counter, idx, batch_idxs, metrics_str)
      summary_str = self.sess.run(metrics_sum)
      self.writer.add_summary(summary_str, counter)

  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      fc2 = linear(tf.reshape(h3, [self.batch_size, -1]), self.y_dim, 'd_fc2_lin')

      return fc2

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
