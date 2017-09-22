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
import inception_v4

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


# the data, shuffled and split between train and test sets
data = np.load('data299.npz')
X_train = data['arr_0']
Y_train = data['arr_1']
X_test = data['arr_2']
Y_test = data['arr_3']

print(Y_train[:,0].shape)
print(Y_test[:,0].shape)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# input image dimensions
img_rows, img_cols = X_train.shape[2], X_train.shape[2]
# the CIFAR10 images are RGB
img_channels = 3

def pop_layer(model):
    if not model.outputs:
        raise Exception('Sequential model cannot be popped: model is empty.')

    model.layers.pop()
    if not model.layers:
        model.outputs = []
        model.inbound_nodes = []
        model.outbound_nodes = []
    else:
        model.layers[-1].outbound_nodes = []
        model.outputs = [model.layers[-1].output]
    model.built = False

# convert class vectors to binary class matrices
# Y_train = np_utils.to_categorical(y_train, nb_classes)
# Y_test = np_utils.to_categorical(y_test, nb_classes)

model = inception_v4.create_model(weights='imagenet')

layer1 = pop_layer(model)
layer2 = pop_layer(model)
layer3 = pop_layer(model)

bottleneck_features_train = model.predict(X_train, 8)
np.save(open('inception_v4_features_train.npy', 'w'), bottleneck_features_train)
bottleneck_features_test = model.predict(X_test, 8)
np.save(open('inception_v4_features_test.npy', 'w'), bottleneck_features_test)
