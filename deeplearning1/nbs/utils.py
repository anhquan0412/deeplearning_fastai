from __future__ import division,print_function
import math, os, json, sys, re

# import cPickle as pickle  # Python 2
import pickle  # Python3

from glob import glob
import numpy as np
from matplotlib import pyplot as plt
from operator import itemgetter, attrgetter, methodcaller
from collections import OrderedDict
import itertools
from itertools import chain

import pandas as pd
import PIL
from PIL import Image
from numpy.random import random, permutation, randn, normal, uniform, choice
from numpy import newaxis
import scipy
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom
from scipy.ndimage import imread
from sklearn.metrics import confusion_matrix
import bcolz
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

from IPython.lib.display import FileLink

import theano
from theano import shared, tensor as T
from theano.tensor.nnet import conv2d, nnet
from theano.tensor.signal import pool

import keras
from keras import backend as K
from keras.utils.data_utils import get_file
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Reshape, merge, LSTM, Bidirectional
from keras.layers import SpatialDropout1D, Concatenate  # Keras2

from keras.layers import TimeDistributed, Activation, SimpleRNN, GRU
from keras.layers.core import Flatten, Dense, Dropout, Lambda

# from keras.regularizers import l2, activity_l2, l1, activity_l1  # Keras1
from keras.regularizers import l2, l1  # Keras2

from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD, RMSprop, Adam

# from keras.utils.layer_utils import layer_from_config  # Keras1
from keras.layers import deserialize  # Keras 2
from keras.layers.merge import dot, add, concatenate  # Keras2
from keras.metrics import categorical_crossentropy, categorical_accuracy
from keras.layers.convolutional import *
from keras.preprocessing import image, sequence
from keras.preprocessing.text import Tokenizer

#from vgg16_tf import *
from vgg16 import *
from vgg16bn import *
np.set_printoptions(precision=4, linewidth=100)



def save_array(fname, arr):
    """
    save numpy array to file
    """
    c=bcolz.carray(arr, rootdir=fname, mode='w')
    c.flush()

def load_array(fname):
    return bcolz.open(fname)[:]

def plots(ims, figsize=(12,6), rows=1, interp=False, titles=None):
	if type(ims[0]) is np.ndarray:
		ims = np.array(ims).astype(np.uint8)
		if (ims.shape[-1] != 3):
			ims = ims.transpose((0,2,3,1))
	f = plt.figure(figsize=figsize)
	for i in range(len(ims)):
		sp = f.add_subplot(rows, len(ims)//rows, i+1)
		sp.axis('Off')
		if titles is not None:
			sp.set_title(titles[i], fontsize=8)
		plt.imshow(ims[i], interpolation=None if interp else 'none')

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	(This function is copied from the scikit docs.)
	"""
	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)

	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	print(cm)
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j], horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

def get_data(path, target_size=(224,224)):
	"""
	Get images (numpy array of pixels) from path
	"""
	batches = get_batches(path, shuffle=False, batch_size=1, class_mode=None, target_size=target_size)

	return np.concatenate([batches.next() for i in range(batches.samples)])  # Keras2

def get_batches(dirname, gen=image.ImageDataGenerator(), shuffle=True, batch_size=4, class_mode='categorical',
                target_size=(224,224)):
	"""
	Similar to VGG16 get_batches: get batch of image (with data augmentation from image data generator) from directory
	"""
	return gen.flow_from_directory(dirname, target_size=target_size,
            class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)
