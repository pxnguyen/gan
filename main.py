import os
import scipy.misc
import numpy as np

from model import DCGAN
from supervised import FullySupervised
from cfmgan import fmGAN
from primal import Primal
from cprimal import CPrimal
from dual import Dual
from utils import pp, visualize, to_json, show_all_variables
import pdb
import socket

import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 1e-4, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("train_size", np.inf, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 108, "The size of image to use (will be center cropped). [108]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 64, "The size of the output images to produce [64]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_integer("c_dim", 3, "Dimension of image color. [3]")
flags.DEFINE_integer("y_dim", 14, "Number of labels. [14]")
flags.DEFINE_string("dataset", "celebA", "The name of dataset [celebA, mnist, lsun]")
flags.DEFINE_string("input_fname_pattern", "*.jpg", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_string("exp_name", 'basic', "Experiment name [basic]")
flags.DEFINE_string("norm", 'l1', "Experiment name [basic]")
flags.DEFINE_string("train_mode", 'all', "True for training with labels, False for not [False]")
flags.DEFINE_string("generator", 'lsgan', "True for training with labels, False for not [False]")
flags.DEFINE_string("discriminator", 'lsgan', "True for training with labels, False for not [False]")
flags.DEFINE_string("version",
        'feature_matching', "Architecture choices [feature_matching, basic, supervised]")
flags.DEFINE_string("data_dir", '/mnt/hermes/nguyenpx/', "Path to the data directories")
flags.DEFINE_boolean("is_train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("is_crop", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("visualize", False, "True for visualizing, False for nothing [False]")
FLAGS = flags.FLAGS

version_list = ['basic', 'feature_matching', 'supervised',
    'wgan', 'wgan2', 'cfm', 'cprimal-clip',
    'primal-clip', 'dual-clip',
    'dual-gp']
if FLAGS.version not in version_list:
    raise Exception('Wrong version.')

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height
  if FLAGS.output_width is None:
    FLAGS.output_width = FLAGS.output_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)

  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)
  sample_dir = os.path.join(FLAGS.sample_dir, FLAGS.exp_name)
  if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

  vis_dir = os.path.join(FLAGS.sample_dir, FLAGS.exp_name, 'vis')
  if not os.path.exists(vis_dir):
    os.makedirs(vis_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
    if FLAGS.version == 'supervised':
      model = FullySupervised(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        y_dim=FLAGS.y_dim,
        c_dim=3,
        dataset_name=FLAGS.dataset,
        exp_name=FLAGS.exp_name,
        input_fname_pattern=FLAGS.input_fname_pattern,
        is_crop=FLAGS.is_crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        version=FLAGS.version)
    elif FLAGS.version == 'cfm':
      model = fmGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=100,
        y_dim=FLAGS.y_dim,
        c_dim=3,
        dataset_name=FLAGS.dataset,
        exp_name=FLAGS.exp_name,
        input_fname_pattern=FLAGS.input_fname_pattern,
        is_crop=FLAGS.is_crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        version=FLAGS.version)
    elif FLAGS.version == 'primal-clip':
      model = Primal(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=100,
        gen_name=FLAGS.generator,
        disc_name=FLAGS.discriminator,
        y_dim=FLAGS.y_dim,
        train_mode=FLAGS.train_mode,
        c_dim=3,
        dataset_name=FLAGS.dataset,
        exp_name=FLAGS.exp_name,
        input_fname_pattern=FLAGS.input_fname_pattern,
        is_crop=FLAGS.is_crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        version=FLAGS.version)
    elif FLAGS.version == 'cprimal-clip':
      model = CPrimal(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=FLAGS.batch_size,
        y_dim=FLAGS.y_dim,
        train_mode=FLAGS.train_mode,
        c_dim=3,
        dataset_name=FLAGS.dataset,
        exp_name=FLAGS.exp_name,
        input_fname_pattern=FLAGS.input_fname_pattern,
        is_crop=FLAGS.is_crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        version=FLAGS.version)

    elif FLAGS.version == 'dual-gp':
      model = Dual(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=100,
        y_dim=FLAGS.y_dim,
        c_dim=3,
        dataset_name=FLAGS.dataset,
        exp_name=FLAGS.exp_name,
        input_fname_pattern=FLAGS.input_fname_pattern,
        is_crop=FLAGS.is_crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        version=FLAGS.version)
    elif FLAGS.version == 'dual-clip':
      model = Dual(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        norm=FLAGS.norm,
        sample_num=100,
        y_dim=FLAGS.y_dim,
        c_dim=3,
        dataset_name=FLAGS.dataset,
        exp_name=FLAGS.exp_name,
        input_fname_pattern=FLAGS.input_fname_pattern,
        is_crop=FLAGS.is_crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        version=FLAGS.version)
    elif FLAGS.version == 'dcgan':
      model = DCGAN(
        sess,
        input_width=FLAGS.input_width,
        input_height=FLAGS.input_height,
        output_width=FLAGS.output_width,
        output_height=FLAGS.output_height,
        batch_size=FLAGS.batch_size,
        sample_num=100,
        y_dim=FLAGS.y_dim,
        c_dim=3,
        dataset_name=FLAGS.dataset,
        exp_name=FLAGS.exp_name,
        input_fname_pattern=FLAGS.input_fname_pattern,
        is_crop=FLAGS.is_crop,
        checkpoint_dir=FLAGS.checkpoint_dir,
        sample_dir=FLAGS.sample_dir,
        version=FLAGS.version)
    else:
      fprintf("Not implemented yet")

    show_all_variables()
    if FLAGS.is_train:
      model.train(FLAGS)
    elif FLAGS.visualize:
      model.visualize(FLAGS)
    else:
      model.eval(FLAGS)

if __name__ == '__main__':
  tf.app.run()
