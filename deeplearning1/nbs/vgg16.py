from __future__ import division, print_function

import os, json
from glob import glob
import numpy as np
from scipy import misc, ndimage
from scipy.ndimage.interpolation import zoom

from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.utils.data_utils import get_file
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Lambda
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D  # Conv2D: Keras2
from keras.layers.pooling import GlobalAveragePooling2D
from keras.optimizers import SGD, RMSprop, Adam
from keras.preprocessing import image


vgg_mean = np.array([123.68, 116.779, 103.939], dtype=np.float32).reshape((3,1,1)) # change to (1,1,3) for tensorflow
#these 3 numbers are means (for each channel, there are 3) of all images in imagenet


def vgg_preprocess(x):
    x = x - vgg_mean # to center the data. Another way: normalize each value to z score (x-mean/std)
    return x[::-1,:,:] # reverse axis rgb->bgr, and change to x[:,:,::-1] for tensorflow


class Vgg16():
    """The VGG 16 Imagenet model"""


    def __init__(self):
        self.FILE_PATH = 'http://files.fast.ai/models/'
        self.create()
        self.get_classes()


    def get_classes(self):
        """
        Get imagenet class json file from http://files.fast.ai/models/imagenet_class_index.json
        Save class list tp self.classes 
        """
        fname = 'imagenet_class_index.json'
        fpath = get_file(fname, self.FILE_PATH+fname, cache_subdir='models')
        with open(fpath) as f:
            class_dict = json.load(f)
        self.classes = [class_dict[str(i)][1] for i in range(len(class_dict))]

    def predict(self, imgs, details=False):
        all_preds = self.model.predict(imgs)
        idxs = np.argmax(all_preds, axis=1)
        preds = [all_preds[i, idxs[i]] for i in range(len(idxs))]
        classes = [self.classes[idx] for idx in idxs]
        return np.array(preds), idxs, classes


    def ConvBlock(self, layers, filters):
        model = self.model
        for i in range(layers):
            model.add(ZeroPadding2D((1, 1)))
            model.add(Conv2D(filters, kernel_size=(3, 3), activation='relu'))  # Keras2
        model.add(MaxPooling2D((2, 2), strides=(2, 2)))


    def FCBlock(self):
        model = self.model
        model.add(Dense(4096, activation='relu'))
        model.add(Dropout(0.5))


    def create(self):
        """
        Create vgg16 model from scratch
        Load weight from http://files.fast.ai/models/vgg16.h5
        Save model to self.model
        """
        model = self.model = Sequential()
        model.add(Lambda(vgg_preprocess, input_shape=(3,224,224), output_shape=(3,224,224))) # change to (224,224,3) for tensorflow

        self.ConvBlock(2, 64)
        self.ConvBlock(2, 128)
        self.ConvBlock(3, 256)
        self.ConvBlock(3, 512)
        self.ConvBlock(3, 512)

        model.add(Flatten())
        self.FCBlock()
        self.FCBlock()
        model.add(Dense(1000, activation='softmax'))

        fname = 'vgg16.h5'
        model.load_weights(get_file(fname, self.FILE_PATH+fname, cache_subdir='models'))


    def get_batches(self, path, gen=image.ImageDataGenerator(), shuffle=True, batch_size=8, class_mode='categorical'):

        """
        Take path to a directory and generates batches of preprocessed data (from ImageDataGenerator)
        Work best when fetched to fit_generator
        """
        return gen.flow_from_directory(path, target_size=(224,224),
                class_mode=class_mode, shuffle=shuffle, batch_size=batch_size)


    def ft(self, num):
        """
        Pop last layers, mark all layers untrainable (since we already got the weight prior to this)
        Then add a dense layers that matches the class label outputs. (We only train this layer)
        """
        model = self.model
        model.pop()
        for layer in model.layers: layer.trainable=False
        model.add(Dense(num, activation='softmax'))
        self.compile()

    def finetune(self, batches):
        """
        Remove last layers and add a dense layers that fits the current class labels
        """
        self.ft(batches.num_class)  # Keras2
        # re-sort
        # ex: {dog:1,cat:0} = > self.classes = [cat,dog]
        classes = list(iter(batches.class_indices))
        for c in batches.class_indices:
            classes[batches.class_indices[c]] = c
        self.classes = classes


    def compile(self, lr=0.001):
        self.model.compile(optimizer=Adam(lr=lr),
                loss='categorical_crossentropy', metrics=['accuracy'])


    # Keras2
    def fit_data(self, trn, labels,  val, val_labels,  nb_epoch=1, batch_size=64):
        self.model.fit(trn, labels, epochs=nb_epoch,
                validation_data=(val, val_labels), batch_size=batch_size)


    # Keras2
    def fit(self, batches, val_batches, batch_size, nb_epoch=1):
        """
        Call fit_generator from keras. Works best with ImageDataGenerator.flow_from_directory
        """
        self.model.fit_generator(batches, steps_per_epoch=int(np.ceil(batches.samples/batch_size)), epochs=nb_epoch,
                validation_data=val_batches, validation_steps=int(np.ceil(val_batches.samples/batch_size)))

        
    # Keras2
    def test(self, path, batch_size=8):
        """
        Get batch of testing data and apply predict_generator
        Return test batches and numpy array of predictions
        """
        test_batches = self.get_batches(path, shuffle=False, batch_size=batch_size, class_mode=None)
        return test_batches, self.model.predict_generator(test_batches, steps=int(np.ceil(test_batches.samples/batch_size)))
