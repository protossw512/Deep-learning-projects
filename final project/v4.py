'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD
from keras.layers.pooling import GlobalAveragePooling2D
from keras.regularizers import l2
from keras.utils import np_utils
import matplotlib.pyplot as plt
import keras
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
import math
import cPickle as pickle
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.applications.resnet50 import ResNet50


plt.ioff()

def step_decay(epoch):
	initial_lrate = 0.01
	drop = 0.1
	epochs_drop = 50.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

batch_size = 32
nb_classes = 29
nb_epoch = 80
data_augmentation = True

# input image dimensions
img_rows, img_cols = 256, 256
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_all = np.load('all_x.npy')
Y_all = np.load('all_y.npy')
X_train = X_all[0:4000, :]
y_train = Y_all[0:4000, 0]
print(y_train.shape)
X_test = X_all[4000:4722, :]
y_test = Y_all[4000:4722, 0]
print(y_test.shape)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model_resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(img_channels, img_rows, img_cols))

bottleneck_features_train = model_resnet.predict(X_train, 16)
np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
bottleneck_features_test = model_resnet.predict(X_test, 16)
np.save(open('bottleneck_features_test.npy', 'w'), bottleneck_features_test)
