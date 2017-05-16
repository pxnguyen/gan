"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import json
import os
import math
import numpy as np
import pdb
import pprint
import random
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim

from time import gmtime, strftime
from six.moves import xrange

pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

def show_all_variables():
  model_vars = tf.trainable_variables()
  slim.model_analyzer.analyze_vars(model_vars, print_info=True)

def get_image(image_path, input_height, input_width,
              resize_height=64, resize_width=64,
              is_crop=True, is_grayscale=False):
  try:
    image = imread(image_path, is_grayscale)
  except Error as E:
    print E

  return transform(image, input_height, input_width,
                   resize_height, resize_width, is_crop)

def save_images(images, size, image_path):
  return imsave(inverse_transform(images), size, image_path)

def load_labels(labels_dict, filenames, label_set_size):
    fns = map(lambda x: os.path.splitext(os.path.split(x)[1])[0], filenames)
    embedding = np.zeros([len(filenames), label_set_size], dtype=np.float32)
    for index, fn in enumerate(fns):
        labels = np.array(labels_dict[fn])-1
        for l in labels:
            embedding[index][l] = 1
    return embedding

def load_labels_tf_format(labels_dict, filenames, label_set_size):
    fns = map(lambda x: os.path.splitext(os.path.split(x)[1])[0], filenames)
    embedding = -1*np.ones([len(filenames), label_set_size], dtype=np.int64)
    for index, fn in enumerate(fns):
        labels = np.array(labels_dict[fn])-1
        for l in labels:
            embedding[index][l] = l
    return embedding

def load_label_dict(data_dir, mode):
    file_path = os.path.join(data_dir, '%s/labels.txt' % mode)
    label_dict = {}
    with open(file_path) as fid:
        lines = fid.read().split('\n')
        for line in lines[:-1]:
            parts = line.split(':')
            name = parts[0]
            print parts
            labels = map(int, parts[1].split(' '))
            label_dict[name] = labels
    return label_dict

def imread(path, is_grayscale = False):
  if (is_grayscale):
    return scipy.misc.imread(path, flatten = True).astype(np.float)
  else:
    return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
  return inverse_transform(images)

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def imsave(images, size, path):
  image = np.squeeze(merge(images, size))
  return scipy.misc.imsave(path, image)

def center_crop(x, crop_h, crop_w,
                resize_h=64, resize_w=64):
  if crop_w is None:
    crop_w = crop_h
  h, w = x.shape[:2]
  j = int(round((h - crop_h)/2.))
  i = int(round((w - crop_w)/2.))
  return scipy.misc.imresize(
      x[j:j+crop_h, i:i+crop_w], [resize_h, resize_w])

def transform(image, input_height, input_width, 
              resize_height=64, resize_width=64, is_crop=True):
  if is_crop:
    cropped_image = center_crop(
      image, input_height, input_width, 
      resize_height, resize_width)
  else:
    cropped_image = scipy.misc.imresize(image, [resize_height, resize_width])
  return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
  return (images+1.)/2.

def to_json(output_path, *layers):
  with open(output_path, "w") as layer_f:
    lines = ""
    for w, b, bn in layers:
      layer_idx = w.name.split('/')[0].split('h')[1]

      B = b.eval()

      if "lin/" in w.name:
        W = w.eval()
        depth = W.shape[1]
      else:
        W = np.rollaxis(w.eval(), 2, 0)
        depth = W.shape[0]

      biases = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(B)]}
      if bn != None:
        gamma = bn.gamma.eval()
        beta = bn.beta.eval()

        gamma = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(gamma)]}
        beta = {"sy": 1, "sx": 1, "depth": depth, "w": ['%.2f' % elem for elem in list(beta)]}
      else:
        gamma = {"sy": 1, "sx": 1, "depth": 0, "w": []}
        beta = {"sy": 1, "sx": 1, "depth": 0, "w": []}

      if "lin/" in w.name:
        fs = []
        for w in W.T:
          fs.append({"sy": 1, "sx": 1, "depth": W.shape[0], "w": ['%.2f' % elem for elem in list(w)]})

        lines += """
          var layer_%s = {
            "layer_type": "fc", 
            "sy": 1, "sx": 1, 
            "out_sx": 1, "out_sy": 1,
            "stride": 1, "pad": 0,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx.split('_')[0], W.shape[1], W.shape[0], biases, gamma, beta, fs)
      else:
        fs = []
        for w_ in W:
          fs.append({"sy": 5, "sx": 5, "depth": W.shape[3], "w": ['%.2f' % elem for elem in list(w_.flatten())]})

        lines += """
          var layer_%s = {
            "layer_type": "deconv", 
            "sy": 5, "sx": 5,
            "out_sx": %s, "out_sy": %s,
            "stride": 2, "pad": 1,
            "out_depth": %s, "in_depth": %s,
            "biases": %s,
            "gamma": %s,
            "beta": %s,
            "filters": %s
          };""" % (layer_idx, 2**(int(layer_idx)+2), 2**(int(layer_idx)+2),
               W.shape[0], W.shape[3], biases, gamma, beta, fs)
    layer_f.write(" ".join(lines.replace("'","").split()))

def make_gif(images, fname, duration=2, true_image=False):
  import moviepy.editor as mpy

  def make_frame(t):
    try:
      x = images[int(len(images)/duration*t)]
    except:
      x = images[-1]

    if true_image:
      return x.astype(np.uint8)
    else:
      return ((x+1)/2*255).astype(np.uint8)

  clip = mpy.VideoClip(make_frame, duration=duration)
  clip.write_gif(fname, fps = len(images) / duration)

def visualize(sess, dcgan, config):
  image_frame_dim = int(math.ceil(config.batch_size**.5))
  z = np.random.normal(0, 10, size=(config.batch_size, dcgan.z_dim))
  for i in range(dcgan.y_dim):
    print i
    y = np.random.choice(14, config.batch_size)
    y_one_hot = np.zeros((config.batch_size, 14))
    y_one_hot[np.arange(config.batch_size), y] = 1

    samples = sess.run(dcgan.sampler, feed_dict={dcgan.z: z, dcgan.y: y_one_hot})
    pdb.set_trace()
    save_images(samples,
        [image_frame_dim, image_frame_dim],
        './samples/test_%s_%d.png' % (strftime("%Y%m%d%H%M%S", gmtime()), i))

    #try:
    #  make_gif(samples, './samples/test_gif_%s.gif' % (idx))
    #except:
    #  save_images(samples, [image_frame_dim, image_frame_dim], './samples/test_%s.png' % strftime("%Y%m%d%H%M%S", gmtime()))
