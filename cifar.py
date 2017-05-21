import cPickle as pickle
import numpy as np
import os
import pdb
import random

def unpickle(filepath):
  fo = open(filepath, 'rb')
  data_lbls = pickle.load(fo)
  fo.close()
  return data_lbls

def cifar_generator_cond(filenames, batch_size, data_dir):
    all_data = []
    all_lbls = []
    for filename in filenames:
        data_lbls = unpickle(data_dir + '/' + filename)
        all_data.append(data_lbls['data'])
        all_lbls.append(data_lbls['labels'])

    images = np.concatenate(all_data, axis=0)
    lbls = np.concatenate(all_lbls, axis=0)
    image_set = []
    for i in range(10):
        indeces = np.where(lbls==i)
        image_set.append(images[indeces])

    def get_epoch():
      for i in xrange(len(images) / batch_size):
        images_toreturn = []
        labels_toreturn = []
        for lbl in range(10):
          images_lbl = image_set[lbl]
          population = range(len(images_lbl))
          indeces = random.sample(population, batch_size/10)
          imgs_part = images_lbl[indeces]
          lbls_part = [lbl for i in range(batch_size/10)]
          images_toreturn.append(imgs_part)
          labels_toreturn += lbls_part
        yield images_toreturn, labels_toreturn

    return get_epoch

def cifar_generator(filenames, batch_size, data_dir):
    all_data = []
    all_lbls = []
    for filename in filenames:
        data_lbls = unpickle(data_dir + '/' + filename)
        all_data.append(data_lbls['data'])
        all_lbls.append(data_lbls['labels'])

    images = np.concatenate(all_data, axis=0)
    lbls = np.concatenate(all_lbls, axis=0)

    def get_epoch():
      population = range(len(images))
      for i in xrange(len(images) / batch_size):
        indeces = random.sample(population, batch_size)
        yield [np.copy(images[indeces]),
            lbls[indeces]]

    return get_epoch

def load(batch_size, data_dir, with_cond=False):
    if with_cond:
      return (
          cifar_generator_cond(['data_batch_1',
              'data_batch_2', 'data_batch_3',
              'data_batch_4','data_batch_5'], batch_size, data_dir),
          cifar_generator(['test_batch'], batch_size, data_dir)
      )
    else:
      return (
          cifar_generator(['data_batch_1',
              'data_batch_2', 'data_batch_3',
              'data_batch_4','data_batch_5'], batch_size, data_dir),
          cifar_generator(['test_batch'], batch_size, data_dir)
      )
