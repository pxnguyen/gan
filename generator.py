from __future__ import division
import cifar
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

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

def generator_began(self, z, y):
    with tf.variable_scope("generator") as scope:
      z = concat([z, y], 1)
      self.z_, self.h0_w, self.h0_b = linear(
          z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)

      self.h0 = tf.reshape(
          self.z_, [-1, 4, 4, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(self.h0))

      self.h1, self.h1_w, self.h1_b = deconv2d(
          h0, [self.batch_size, 8, 8, self.gf_dim*4],
          name='g_h1', with_w=True)
      h1 = tf.nn.relu(self.g_bn1(self.h1))

      h2, self.h2_w, self.h2_b = deconv2d(
          h1, [self.batch_size, 16, 16, self.gf_dim*2],
          name='g_h2', with_w=True)
      h2 = tf.nn.relu(self.g_bn2(h2))

      h3, self.h3_w, self.h3_b = deconv2d(
          h2, [self.batch_size, 32, 32, 3],
          name='g_h3', with_w=True)
      h3 = tf.nn.relu(self.g_bn3(h3))

      return tf.nn.tanh(h3)

def sampler_began(self, z, y):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()
      z = concat([z, y], 1)
      self.z_, self.h0_w, self.h0_b = linear(
          z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)

      self.h0 = tf.reshape(
          self.z_, [-1, 4, 4, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(self.h0))

      self.h1, self.h1_w, self.h1_b = deconv2d(
          h0, [self.sample_num, 8, 8, self.gf_dim*4],
          name='g_h1', with_w=True)
      h1 = tf.nn.relu(self.g_bn1(self.h1))

      h2, self.h2_w, self.h2_b = deconv2d(
          h1, [self.sample_num, 16, 16, self.gf_dim*2],
          name='g_h2', with_w=True)
      h2 = tf.nn.relu(self.g_bn2(h2))

      h3, self.h3_w, self.h3_b = deconv2d(
          h2, [self.sample_num, 32, 32, 3],
          name='g_h3', with_w=True)
      h3 = tf.nn.relu(self.g_bn3(h3))

      return tf.nn.tanh(h3)

def generator_dcgan(self, z, y):
    with tf.variable_scope("generator") as scope:
      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      z = concat([z, y], 1)
      self.z_, self.h0_w, self.h0_b = linear(
          z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

      self.h0 = tf.reshape(
          self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(self.h0))

      self.h1, self.h1_w, self.h1_b = deconv2d(
          h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4],
          name='g_h1', with_w=True)
      h1 = tf.nn.relu(self.g_bn1(self.h1))

      h2, self.h2_w, self.h2_b = deconv2d(
          h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2],
          name='g_h2', with_w=True)
      h2 = tf.nn.relu(self.g_bn2(h2))

      h3, self.h3_w, self.h3_b = deconv2d(
          h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1],
          name='g_h3', with_w=True)
      h3 = tf.nn.relu(self.g_bn3(h3))

      h4, self.h4_w, self.h4_b = deconv2d(
          h3, [self.batch_size, s_h, s_w, self.c_dim],
          name='g_h4', with_w=True)

      return tf.nn.tanh(h4)

def sampler_dcgan(self, z, y):
    with tf.variable_scope("generator") as scope:
      scope.reuse_variables()

      s_h, s_w = self.output_height, self.output_width
      s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
      s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
      s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
      s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

      # project `z` and reshape
      z = concat([z, y], 1)
      h0 = tf.reshape(
          linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
          [-1, s_h16, s_w16, self.gf_dim * 8])
      h0 = tf.nn.relu(self.g_bn0(h0, train=False))

      h1 = deconv2d(h0, [self.sample_num, s_h8, s_w8, self.gf_dim*4], name='g_h1')
      h1 = tf.nn.relu(self.g_bn1(h1, train=False))

      h2 = deconv2d(h1, [self.sample_num, s_h4, s_w4, self.gf_dim*2], name='g_h2')
      h2 = tf.nn.relu(self.g_bn2(h2, train=False))

      h3 = deconv2d(h2, [self.sample_num, s_h2, s_w2, self.gf_dim*1], name='g_h3')
      h3 = tf.nn.relu(self.g_bn3(h3, train=False))

      h4 = deconv2d(h3, [self.sample_num, s_h, s_w, self.c_dim], name='g_h4')

      return tf.nn.tanh(h4)

def generator_cifar(self, z, y):
  with tf.variable_scope("generator") as scope:
    dim = 64
    bs = self.batch_size
    z = tf.concat(values=[z, y], axis=1)
    h0 = linear(z, 4*4*4*dim, 'g_h0_lin')
    h0 = self.g_bn0(h0)
    h0 = tf.nn.relu(h0)
    h0 = tf.reshape(h0, [bs, 4, 4, 4*dim])

    h1 = deconv2d(h0, [bs, 8, 8, dim*2], name='g_h1')
    h1 = self.g_bn1(h1)
    h1 = tf.nn.relu(h1)

    h2 = deconv2d(h1, [bs, 16, 16, dim], name='g_h2')
    h2 = self.g_bn2(h2)
    h2 = tf.nn.relu(h2)

    h3 = deconv2d(h2, [bs, 32, 32, 3], name='g_h3')
    h3 = tf.nn.tanh(h3)
    return h3

def sampler_cifar(self, z, y):
  with tf.variable_scope("generator") as scope:
    scope.reuse_variables()
    dim = 64
    bs = self.sample_num
    z = tf.concat(values=[z, y], axis=1)
    h0 = linear(z, 4*4*4*dim, 'g_h0_lin')
    h0 = self.g_bn0(h0, train=False)
    h0 = tf.nn.relu(h0)
    h0 = tf.reshape(h0, [bs, 4, 4, 4*dim])

    h1 = deconv2d(h0, [bs, 8, 8, dim*2], name='g_h1')
    h1 = self.g_bn1(h1, train=False)
    h1 = tf.nn.relu(h1)

    h2 = deconv2d(h1, [bs, 16, 16, dim], name='g_h2')
    h2 = self.g_bn2(h2, train=False)
    h2 = tf.nn.relu(h2)

    h3 = deconv2d(h2, [bs, 32, 32, 3], name='g_h3')
    h3 = tf.nn.tanh(h3)
    return h3

def generator_mnist(self, z, y, config):
    z = tf.concat(values=[z, y], axis=1)

    bs = self.batch_size
    dim = 64
    yb = tf.reshape(y, [bs, 1, 1, config.y_dim])
    h0 = linear(z, 4*4*4*dim, 'g_h0_lin')
    #h0 = self.g_bn0(h0)
    h0 = tf.nn.relu(h0)
    h0 = tf.reshape(h0, [-1, 4, 4, 4*dim])
    h0 = conv_cond_concat(h0, yb)

    h1 = deconv2d(h0, [bs, 8, 8, dim*2], name='g_h1')
    #h1 = self.g_bn1(h1)
    h1 = tf.nn.relu(h1)
    h1 = conv_cond_concat(h1, yb)

    h2 = deconv2d(h1, [bs, 16, 16, dim], name='g_h2')
    #h2 = self.g_bn2(h2)
    h2 = tf.nn.relu(h2)
    h2 = conv_cond_concat(h2, yb)

    h3 = deconv2d(h2, [bs, 28, 28, 3], name='g_h3')
    h3 = tf.nn.tanh(h3)
    return h3
