from __future__ import division
import math
import numpy as np
import os
import pdb
import tensorflow as tf
import time

from six.moves import xrange
from glob import glob
from ops import *
from utils import *
from tensorflow.contrib.metrics import streaming_precision_at_thresholds

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):
  def __init__(self, sess, input_height=108, input_width=108, is_crop=True,
         batch_size=64, sample_num = 64, output_height=64, output_width=64,
         y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
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
    self.build_model()

  def build_model(self):
    self.y = tf.placeholder(tf.float32, [self.batch_size, self.y_dim], name='y')

    if self.is_crop:
      image_dims = [self.output_height, self.output_width, self.c_dim]
    else:
      image_dims = [self.input_height, self.input_width, self.c_dim]

    self.inputs = tf.placeholder(
      tf.float32, [self.batch_size] + image_dims, name='real_images')
    self.sample_inputs = tf.placeholder(
      tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

    inputs = self.inputs
    sample_inputs = self.sample_inputs

    self.z = tf.placeholder(
      tf.float32, [None, self.z_dim], name='z')
    self.z_sum = histogram_summary("z", self.z)
    self.G = self.generator(self.z, self.y)
    self.D, self.D_logits, self.real_cat_logits, self.h3_real = self.discriminator(inputs, self.y, reuse=False)
    self.h3_real = tf.reduce_mean(self.h3_real, axis=0)

    self.sampler = self.sampler(self.z, self.y)
    self.D_, self.D_logits_, self.fake_cat_logits, self.h3_fake = self.discriminator(self.G, self.y, reuse=True)
    self.h3_fake = tf.reduce_mean(self.h3_fake, axis=0)

    self.d_sum = histogram_summary("d", self.D)
    self.d__sum = histogram_summary("d_", self.D_)
    self.inputs_sum = image_summary("inputs", inputs)
    self.G_sum = image_summary("G", self.G)

    def sigmoid_cross_entropy_with_logits(x, y):
        return tf.nn.sigmoid_cross_entropy_with_logits(logits=x, labels=y)

    self.category_loss_real = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
          logits=self.real_cat_logits,
          labels=self.y))

    self.category_loss_fake = tf.reduce_mean(
        tf.nn.sigmoid_cross_entropy_with_logits(
          logits=self.fake_cat_logits,
          labels=self.y))

    self.d_loss_real = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits, 0.9*tf.ones_like(self.D)))
    self.d_loss_fake = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
    self.g_loss_gan = tf.reduce_mean(
      sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))

    self.feature_matching_loss = tf.nn.l2_loss(self.h3_real - self.h3_fake)
    self.fm_loss_sum = scalar_summary("feature_matching_loss", self.feature_matching_loss)

    self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
    self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
    self.category_loss_real_sum = scalar_summary("category_loss_real", self.category_loss_real)
    self.category_loss_fake_sum = scalar_summary("category_loss_fake", self.category_loss_fake)

    self.d_loss = self.d_loss_real + self.d_loss_fake\
        + self.category_loss_real

    if self.exp_name == 'feature_matching':
      self.g_loss = self.feature_matching_loss + self.category_loss_fake
    else:
      self.g_loss = self.g_loss_gan + self.category_loss_fake

    self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
    self.d_loss_sum = scalar_summary("d_loss", self.d_loss)

    t_vars = tf.trainable_variables()

    self.d_vars = [var for var in t_vars if 'd_' in var.name]
    self.g_vars = [var for var in t_vars if 'g_' in var.name]

    self.saver = tf.train.Saver()

  def train(self, config):
    """Train DCGAN"""
    if config.dataset == 'mnist':
      data_X, data_y = self.load_mnist()
    elif config.dataset == 'nuswide':
      label_dict = load_label_dict('train')
      data_X = glob(os.path.join("/mnt/hermes/nguyenpx/nuswide/train", self.input_fname_pattern))
      #data_X = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
      data_y = load_labels(label_dict, data_X, 14)
    else:
      data = glob(os.path.join("./data", config.dataset, self.input_fname_pattern))
    #np.random.shuffle(data)

    d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_loss, var_list=self.d_vars)
    g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
    try:
      tf.global_variables_initializer().run()
    except:
      tf.initialize_all_variables().run()

    self.g_sum = merge_summary([self.z_sum, self.d__sum,
      self.G_sum, self.d_loss_fake_sum, self.g_loss_sum, self.category_loss_fake_sum,
      self.fm_loss_sum])
    self.d_sum = merge_summary(
        [self.z_sum, self.d_sum, self.d_loss_real_sum,
          self.d_loss_sum, self.inputs_sum, self.category_loss_real_sum])
    self.writer = SummaryWriter(os.path.join("./logs",
        '{0}_{1}'.format(config.dataset, config.exp_name)), self.sess.graph)

    #TODO(phucng): np.random.normal(0, 1, size=(self.sample_num self.z_dim))
    sample_z = np.random.uniform(-1, 1, size=(self.sample_num , self.z_dim))

    if config.dataset == 'mnist':
      sample_inputs = data_X[0:self.sample_num]
      sample_labels = data_y[0:self.sample_num]
    elif config.dataset == 'nuswide':
      sample_files = data_X[0:self.sample_num]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    is_crop=self.is_crop,
                    is_grayscale=self.is_grayscale) for sample_file in sample_files]
      sample_inputs = np.array(sample).astype(np.float32)
      sample_labels = data_y[0:self.sample_num]
    else:
      sample_files = data[0:self.sample_num]
      sample = [
          get_image(sample_file,
                    input_height=self.input_height,
                    input_width=self.input_width,
                    resize_height=self.output_height,
                    resize_width=self.output_width,
                    is_crop=self.is_crop,
                    is_grayscale=self.is_grayscale) for sample_file in sample_files]
      sample_inputs = np.array(sample).astype(np.float32)
  
    counter = 1
    start_time = time.time()
    ckp_dir = os.path.join(config.checkpoint_dir, config.exp_name)
    could_load, checkpoint_counter = self.load(ckp_dir)
    if could_load:
      counter = checkpoint_counter
      print(" [*] Load SUCCESS")
    else:
      print(" [!] Load failed...")

    for epoch in xrange(config.epoch):
      if config.dataset == 'mnist':
        batch_idxs = min(len(data_X), config.train_size) // config.batch_size
      elif config.dataset == 'nuswide':
        batch_idxs = min(len(data_X), config.train_size) // config.batch_size
      else:
        data = glob(os.path.join(
          "./data", config.dataset, self.input_fname_pattern))
        batch_idxs = min(len(data), config.train_size) // config.batch_size

      for idx in xrange(0, batch_idxs):
        if config.dataset == 'nuswide':
          batch_files = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        is_crop=self.is_crop,
                        is_grayscale=False) for batch_file in batch_files]

          batch_images = np.array(batch).astype(np.float32)
          batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]
        else:
          batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
          batch = [
              get_image(batch_file,
                        input_height=self.input_height,
                        input_width=self.input_width,
                        resize_height=self.output_height,
                        resize_width=self.output_width,
                        is_crop=self.is_crop,
                        is_grayscale=self.is_grayscale) for batch_file in batch_files]
          batch_images = np.array(batch).astype(np.float32)

        batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
              .astype(np.float32)

        if config.dataset == 'nuswide':
          # Update D network
          _, summary_str = self.sess.run([d_optim, self.d_sum],
            feed_dict={ 
              self.inputs: batch_images,
              self.z: batch_z,
              self.y:batch_labels,
            })
          self.writer.add_summary(summary_str, counter)

          # Update G network
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
              self.z: batch_z,
              self.y: batch_labels,
              self.inputs: batch_images,
            })
          self.writer.add_summary(summary_str, counter)

          # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
          _, summary_str = self.sess.run([g_optim, self.g_sum],
            feed_dict={
              self.z: batch_z,
              self.y:batch_labels,
              self.inputs: batch_images,
              })
          self.writer.add_summary(summary_str, counter)

          errD_fake = self.d_loss_fake.eval({
              self.z: batch_z,
              self.y:batch_labels
          })
          errD_real = self.d_loss_real.eval({
              self.inputs: batch_images,
              self.y:batch_labels
          })
          errG = self.g_loss.eval({
              self.z: batch_z,
              self.inputs: batch_images,
              self.y: batch_labels
          })

        counter += 1
        print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
          % (epoch, idx, batch_idxs,
            time.time() - start_time, errD_fake+errD_real, errG))

        if np.mod(counter, 100) == 1:
          samples, d_loss, g_loss = self.sess.run(
            [self.sampler, self.d_loss, self.g_loss],
            feed_dict={
                self.z: sample_z,
                self.inputs: sample_inputs,
                self.y:sample_labels,
            }
          )
          manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
          manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
          save_images(samples, [manifold_h, manifold_w],
                './{}/{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, config.exp_name,
                    epoch, idx))
          print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))

        if np.mod(counter, 200) == 2:
          ckp_dir = os.path.join(config.checkpoint_dir, config.exp_name)
          self.save(ckp_dir, counter)

  def eval(self, config):
    """Eval DCGAN"""
    label_dict = load_label_dict('eval')
    data_X = glob(os.path.join("/mnt/hermes/nguyenpx/nuswide/eval", self.input_fname_pattern))
    data_y = load_labels(label_dict, data_X, 14)

    self.writer = SummaryWriter(os.path.join("./logs",
        '{0}_{1}_eval'.format(config.dataset, config.exp_name)), self.sess.graph)

    batch_idxs = min(len(data_X), config.train_size) // config.batch_size
    _, _, cat_scores, _ = self.discriminator(self.inputs, reuse=True)
    cat_logits = tf.nn.sigmoid(cat_scores)
    precision, update_op = streaming_precision_at_thresholds(cat_logits, self.y, [0.2])
    prec_sum = scalar_summary("precision", precision[0])

    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run()

    done = False
    last_counter = -1
    while not done:
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

      for idx in xrange(0, batch_idxs):
        batch_files = data_X[idx*config.batch_size:(idx+1)*config.batch_size]
        batch = [
            get_image(batch_file,
                      input_height=self.input_height,
                      input_width=self.input_width,
                      resize_height=self.output_height,
                      resize_width=self.output_width,
                      is_crop=self.is_crop,
                      is_grayscale=False) for batch_file in batch_files]

        batch_images = np.array(batch).astype(np.float32)
        batch_labels = data_y[idx*config.batch_size:(idx+1)*config.batch_size]

        precision_value, probs = self.sess.run([update_op, cat_logits], feed_dict={
          self.inputs: batch_images, self.y: batch_labels})

        print 'Eval checkpoint %d [%4d/%4d] prec: %0.4f' % (counter, idx, batch_idxs, precision_value)
      summary_str, precision_value = self.sess.run([prec_sum, precision])
      self.writer.add_summary(summary_str, counter)

  def discriminator(self, image, y=None, reuse=False):
    with tf.variable_scope("discriminator") as scope:
      if reuse:
        scope.reuse_variables()

      h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
      h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*2, name='d_h1_conv')))
      h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, name='d_h2_conv')))
      h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*8, name='d_h3_conv')))
      fc1 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_fc1_lin')
      fc2 = linear(tf.reshape(h3, [self.batch_size, -1]), self.y_dim, 'd_fc2_lin')

      return tf.nn.sigmoid(fc1), fc1, fc2, h3

  def generator(self, z, y=None):
    with tf.variable_scope("generator") as scope:
        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        z = tf.concat([z, y], 1)

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(
            z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(
            self.z_, [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv2d(
            h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv2d(
            h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv2d(
            h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv2d(
            h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4', with_w=True)

        return tf.nn.tanh(h4)

  def sampler(self, z, y=None):
    with tf.variable_scope("generator") as scope:
        scope.reuse_variables()

        s_h, s_w = self.output_height, self.output_width
        s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
        s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
        s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
        s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

        z = tf.concat([z, y], 1);

        # project `z` and reshape
        h0 = tf.reshape(
            linear(z, self.gf_dim*8*s_h16*s_w16, 'g_h0_lin'),
            [-1, s_h16, s_w16, self.gf_dim * 8])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')

        return tf.nn.tanh(h4)

  def load_mnist(self):
    data_dir = os.path.join("./data", self.dataset_name)
    
    fd = open(os.path.join(data_dir,'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trX = loaded[16:].reshape((60000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    trY = loaded[8:].reshape((60000)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teX = loaded[16:].reshape((10000,28,28,1)).astype(np.float)

    fd = open(os.path.join(data_dir,'t10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd,dtype=np.uint8)
    teY = loaded[8:].reshape((10000)).astype(np.float)

    trY = np.asarray(trY)
    teY = np.asarray(teY)
    
    X = np.concatenate((trX, teX), axis=0)
    y = np.concatenate((trY, teY), axis=0).astype(np.int)
    
    seed = 547
    np.random.seed(seed)
    np.random.shuffle(X)
    np.random.seed(seed)
    np.random.shuffle(y)

    y_vec = np.zeros((len(y), self.y_dim), dtype=np.float)
    for i, label in enumerate(y):
      y_vec[i,y[i]] = 1.0

    return X/255.,y_vec

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
