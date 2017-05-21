from __future__ import division
import cifar
import generator
import math
import numpy as np
import os
import re
import pdb
import socket
import tensorflow as tf
import time
import generator

from six.moves import xrange
from glob import glob
from ops import *
from utils import *
from tensorflow.contrib import metrics

TOWER_NAME = 'tower'

def disc_3layer(self, image, reuse=False):
  with tf.variable_scope("discriminator") as scope:
    if reuse:
      scope.reuse_variables()

    h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
    h1 = conv2d(h0, self.df_dim*2, name='d_h1_conv')
    h1 = lrelu(self.d_bn1_1(h1))

    h2 = conv2d(h1, self.df_dim*4, name='d_h2_conv')
    h2 = lrelu(self.d_bn2_1(h2))

    #h3 = conv2d(h2, self.df_dim*4, name='d_h3_conv')
    #h3 = lrelu(self.d_bn3(h3))

    fc1 = linear(tf.reshape(h2, [self.batch_size, -1]), 1, 'd_fc1_lin')
    fc2 = linear(tf.reshape(h2, [self.batch_size, -1]), self.y_dim, 'fc2_lin')

    return h1, h2, fc1, fc2

def disc_4layer(self, image, reuse=False):
  with tf.variable_scope("discriminator") as scope:
    if reuse:
      scope.reuse_variables()

    h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
    h1_1 = lrelu(self.d_bn1_1(
      conv2d(h0, self.df_dim*2, k_h=3, k_w=3,
        d_h=1, d_w=1, name='d_h1_1_conv')))
    h1_2 = lrelu(self.d_bn1_2(
      conv2d(h1_1, self.df_dim*2, k_h=3, k_w=3,
        d_h=1, d_w=1, name='d_h1_2_conv')))
    h1_3 = lrelu(self.d_bn1_3(
      conv2d(h1_2, self.df_dim*2, k_h=4, k_w=4,
        d_h=2, d_w=2, name='d_h1_3_conv')))

    h2_1 = lrelu(self.d_bn2_1(
      conv2d(h1_3, self.df_dim*4, k_h=3, k_w=3,
        d_h=1, d_w=1, name='d_h2_1_conv')))
    h2_2 = lrelu(self.d_bn2_2(
      conv2d(h2_1, self.df_dim*4, k_h=4, k_w=4,
        d_h=2, d_w=2, name='d_h2_2_conv')))

    h3 = conv2d(h2_2, self.df_dim*8, name='d_h3_conv')
    h3 = lrelu(self.d_bn3_1(h3))

    fc1 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_fc1_lin')
    fc2 = linear(tf.reshape(h3, [self.batch_size, -1]), self.y_dim, 'fc2_lin')

    return h2_2, h3, fc1, fc2

def disc_5layer(self, image, reuse=False):
  with tf.variable_scope("discriminator") as scope:
    if reuse:
      scope.reuse_variables()

    h1_1 = lrelu(self.d_bn1_1(
      conv2d(image, self.df_dim*2, k_h=4, k_w=4,
        d_h=1, d_w=1, name='d_h1_1_conv')))
    h1_2 = lrelu(self.d_bn1_2(
      conv2d(h1_1, self.df_dim*2, k_h=4, k_w=4,
        d_h=2, d_w=2, name='d_h1_2_conv')))
    #16x16

    h2_1 = lrelu(self.d_bn2_1(
      conv2d(h1_2, self.df_dim*4, k_h=4, k_w=4,
        d_h=1, d_w=1, name='d_h2_1_conv')))
    h2_2 = lrelu(self.d_bn2_2(
      conv2d(h2_1, self.df_dim*4, k_h=4, k_w=4,
        d_h=2, d_w=2, name='d_h2_2_conv')))
    #8x8

    h3_1 = lrelu(self.d_bn3_1(
      conv2d(h2_2, self.df_dim*4, k_h=4, k_w=4,
        d_h=1, d_w=1, name='d_h3_1_conv')))
    h3_2 = lrelu(self.d_bn3_2(
      conv2d(h3_1, self.df_dim*4, k_h=1, k_w=4,
        d_h=2, d_w=2, name='d_h3_3_conv')))
    #4x4

    fc1 = linear(tf.reshape(h3_2, [self.batch_size, -1]), 1, 'd_fc1_lin')
    fc2 = linear(tf.reshape(h3_2, [self.batch_size, -1]), self.y_dim, 'fc2_lin')

    return h2_2, h3_2, fc1, fc2

def disc_lsgan(self, image, reuse=False):
  with tf.variable_scope("discriminator") as scope:
    if reuse:
      scope.reuse_variables()

    h1_1 = lrelu(self.d_bn1_1(
      conv2d(image, self.df_dim*2, k_h=3,
        k_w=3, d_h=1, d_w=1, name='d_h1_1_conv')))
    h1_2 = lrelu(self.d_bn1_1(
      conv2d(h1_1, self.df_dim*2, k_h=3,
        k_w=3, d_h=1, d_w=1, name='d_h1_2_conv')))
    h1_3 = lrelu(self.d_bn1_2(
      conv2d(h1_2, self.df_dim*2, k_h=4,
        k_w=4, d_h=2, d_w=2, name='d_h1_3_conv')))
    #16x16

    h2_1 = lrelu(self.d_bn2_1(
      conv2d(h1_3, self.df_dim*4, k_h=3, k_w=3,
        d_h=1, d_w=1, name='d_h2_1_conv')))
    h2_2 = lrelu(self.d_bn2_1(
      conv2d(h2_1, self.df_dim*4, k_h=3, k_w=3,
        d_h=1, d_w=1, name='d_h2_2_conv')))
    h2_3 = lrelu(self.d_bn2_2(
      conv2d(h2_2, self.df_dim*4, k_h=4, k_w=4,
        d_h=2, d_w=2, name='d_h2_3_conv')))
    #8x8

    h3_1 = lrelu(self.d_bn3_1(
      conv2d(h2_3, self.df_dim*4, k_h=3, k_w=3,
        d_h=1, d_w=1, name='d_h3_1_conv')))
    h3_2 = lrelu(self.d_bn3_1(
      conv2d(h3_1, self.df_dim*4, k_h=3, k_w=3,
        d_h=1, d_w=1, name='d_h3_2_conv')))
    h3_3 = lrelu(self.d_bn3_2(
      conv2d(h3_2, self.df_dim*4, k_h=1, k_w=4,
        d_h=2, d_w=2, name='d_h3_3_conv')))
    #4x4

    fc1 = linear(tf.reshape(h3_3, [self.batch_size, -1]), 1, 'd_fc1_lin')
    fc2 = linear(tf.reshape(h3_3, [self.batch_size, -1]), self.y_dim, 'fc2_lin')

    return h2_3, h3_3, fc1, fc2

def disc_began(self, image, reuse=False):
  with tf.variable_scope("discriminator") as scope:
    if reuse:
      scope.reuse_variables()

    h1_1 = lrelu(self.d_bn1_1(
      conv2d(image, self.df_dim*2, k_h=3, k_w=3,
          d_h=1, d_w=1, name='d_h1_1_conv')))
    h1_2 = lrelu(self.d_bn1_2(
      conv2d(h1_1, self.df_dim*2, k_h=3, k_w=3,
          d_h=2, d_w=2, name='d_h1_2_conv')))
    #16x16

    h2_1 = lrelu(self.d_bn2_1(
      conv2d(h1_2, self.df_dim*4, k_h=3, k_w=3,
        d_h=1, d_w=1, name='d_h2_1_conv')))
    h2_2 = lrelu(self.d_bn2_2(
      conv2d(h2_1, self.df_dim*4, k_h=3, k_w=3,
        d_h=2, d_w=2, name='d_h2_2_conv')))
    #8x8

    h3_1 = lrelu(self.d_bn3_1(
      conv2d(h2_2, self.df_dim*4, k_h=3, k_w=3,
        d_h=1, d_w=1, name='d_h3_1_conv')))
    h3_2 = lrelu(self.d_bn3_1(
      conv2d(h3_1, self.df_dim*4, k_h=1, k_w=1,
        d_h=1, d_w=1, name='d_h3_2_conv')))
    #8x8

    fc1 = linear(tf.reshape(h3_2, [self.batch_size, -1]), 1, 'd_fc1_lin')
    fc2 = linear(tf.reshape(h3_2, [self.batch_size, -1]), self.y_dim, 'fc2_lin')

    return h2_2, h3_2, fc1, fc2

def disc_optimal(self, images, reuse=False):
  """Build the CIFAR-10 model.
  Args:
    images: Images returned from distorted_inputs() or inputs().
  Returns:
    Logits.
  """
  NUM_CLASSES = self.y_dim
  with tf.variable_scope('discriminator') as outside_scope:
    if reuse:
      outside_scope.reuse_variables()
    # conv1
    with tf.variable_scope('conv1') as scope:
      kernel = _variable_with_weight_decay(
          'weights',
          shape=[5, 5, 3, 64],
          stddev=5e-2,
          wd=0.0)
      conv = tf.nn.conv2d(images, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.0))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv1 = tf.nn.relu(pre_activation, name=scope.name)
    pdb.set_trace()

    # pool1
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                           padding='SAME', name='pool1')
    # norm1
    norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm1')

    # conv2
    with tf.variable_scope('conv2') as scope:
      kernel = _variable_with_weight_decay('weights',
                                           shape=[5, 5, 64, 64],
                                           stddev=5e-2,
                                           wd=0.0)
      conv = tf.nn.conv2d(norm1, kernel, [1, 1, 1, 1], padding='SAME')
      biases = _variable_on_cpu('biases', [64], tf.constant_initializer(0.1))
      pre_activation = tf.nn.bias_add(conv, biases)
      conv2 = tf.nn.relu(pre_activation, name=scope.name)

    # norm2
    norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                      name='norm2')
    # pool2
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                           strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    # local3
    with tf.variable_scope('local3') as scope:
      # Move everything into depth so we can perform a single matrix multiply.
      reshape = tf.reshape(pool2, [self.batch_size, -1])
      dim = reshape.get_shape()[1].value
      weights = _variable_with_weight_decay('weights', shape=[dim, 384],
                                            stddev=0.04, wd=0.004)
      biases = _variable_on_cpu('biases', [384], tf.constant_initializer(0.1))
      local3 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    # local4
    with tf.variable_scope('local4') as scope:
      weights = _variable_with_weight_decay('weights', shape=[384, 192],
                                            stddev=0.04, wd=0.004)
      biases = _variable_on_cpu('biases', [192], tf.constant_initializer(0.1))
      local4 = tf.nn.relu(tf.matmul(local3, weights) + biases, name=scope.name)

    # linear layer(WX + b),
    with tf.variable_scope('softmax_linear') as scope:
      weights = _variable_with_weight_decay('weights', [192, NUM_CLASSES],
                                            stddev=1/192.0, wd=0.0)
      biases = _variable_on_cpu('biases', [NUM_CLASSES],
                                tf.constant_initializer(0.0))
      softmax_linear = tf.add(tf.matmul(local4, weights), biases, name=scope.name)

  return pool2, local3, local4, softmax_linear

def _variable_with_weight_decay(name, shape, stddev, wd):
  """Helper to create an initialized Variable with weight decay.
  Note that the Variable is initialized with a truncated normal distribution.
  A weight decay is added only if one is specified.
  Args:
    name: name of the variable
    shape: list of ints
    stddev: standard deviation of a truncated Gaussian
    wd: add L2Loss weight decay multiplied by this float. If None, weight
        decay is not added for this Variable.
  Returns:
    Variable Tensor
  """
  dtype = tf.float32
  var = _variable_on_cpu(
      name,
      shape,
      tf.truncated_normal_initializer(stddev=stddev, dtype=dtype))
  if wd is not None:
    weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
  return var

def _variable_on_cpu(name, shape, initializer):
  """Helper to create a Variable stored on CPU memory.
  Args:
    name: name of the variable
    shape: list of ints
    initializer: initializer for Variable
  Returns:
    Variable Tensor
  """
  with tf.device('/cpu:0'):
    dtype = tf.float32
    var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
  return var
  
