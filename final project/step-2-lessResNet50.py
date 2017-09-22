'''Train a simple deep CNN on the CIFAR10 small images dataset.

GPU run command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cifar10_cnn.py

It gets down to 0.65 test logloss in 25 epochs, and down to 0.55 after 50 epochs.
(it's still underfitting at that point, though).

Note: the data was pickled with Python 2, and some encoding issues might prevent you
from loading it in Python 3. You might have to load it in Python 2,
save it in a different format, load it in Python 3 and repickle it.
'''

from __future__ import print_function, absolute_import
import warnings
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.layers.pooling import GlobalAveragePooling2D,GlobalMaxPooling2D
from keras.regularizers import l2
from keras.utils import np_utils
#import matplotlib.pyplot as plt
import keras
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
import math
import cPickle as pickle
import numpy as np
from keras.models import Model
from keras.layers import Input,merge

from keras import backend as K
from keras.engine.topology import get_source_inputs
from keras.utils import layer_utils
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions, preprocess_input, _obtain_input_shape


from keras.applications.resnet50 import identity_block, conv_block

def ResNet50(include_top=True, input_tensor=None, input_shape=None,
             classes=29):
	# Determine proper input shape
	input_shape = _obtain_input_shape(input_shape,
									default_size=224,
									min_size=197,
									dim_ordering=K.image_dim_ordering(),
									include_top=include_top)

	if input_tensor is None:
		img_input = Input(shape=input_shape)
	else:
		if not K.is_keras_tensor(input_tensor):
			img_input = Input(tensor=input_tensor, shape=input_shape)
		else:
			img_input = input_tensor
	if K.image_dim_ordering() == 'tf':
		bn_axis = 3
	else:
		bn_axis = 1

	x = ZeroPadding2D((3, 3))(img_input)
	x = Convolution2D(64, 7, 7, subsample=(2, 2), name='conv1')(x)
	x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
	x = Activation('relu')(x)
	x = MaxPooling2D((3, 3), strides=(2, 2))(x)

	x = conv_block(x, 3, [64, 64, 256], stage=2, block='a1', strides=(1, 1))
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='b1')
	x = identity_block(x, 3, [64, 64, 256], stage=2, block='c1')


	# x = conv_block(x, 3, [64, 64, 256], stage=2, block='a2', strides=(1, 1))
	# x = identity_block(x, 3, [64, 64, 256], stage=2, block='b2')
	# x = identity_block(x, 3, [64, 64, 256], stage=2, block='c2')

	# x = conv_block(x, 3, [64, 64, 256], stage=2, block='a3', strides=(1, 1))
	# x = identity_block(x, 3, [64, 64, 256], stage=2, block='b3')
	# x = identity_block(x, 3, [64, 64, 256], stage=2, block='c3')
    x = AveragePooling2D((7, 7), name='avg_pool')(x)

	if include_top:
		x = Flatten()(x)
		x = Dense(classes, activation='softmax', name='fc1000')(x)

	# Ensure that the model takes into account
	# any potential predecessors of `input_tensor`.
	if input_tensor is not None:
		inputs = get_source_inputs(input_tensor)
	else:
		inputs = img_input
	# Create model.
	model = Model(inputs, x, name='resnet50')
	return model

#plt.ioff()

def step_decay(epoch):
	initial_lrate = 0.01
	drop = 0.15
	epochs_drop = 40.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

batch_size = 32
nb_classes = 29
nb_epoch = 80
data_augmentation = False

# the data, shuffled and split between train and test sets
data = np.load('data224.npz')
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
img_rows, img_cols = X_train.shape[2], X_train.shape[3]
# the CIFAR10 images are RGB
img_channels = 3

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train[:, 0], nb_classes)
Y_test = np_utils.to_categorical(Y_test[:, 0], nb_classes)


'''
model = Sequential()

model.add(BatchNormalization(input_shape=X_train.shape[1:]))
model.add(Flatten())
model.add(Dense(1024, init='he_normal'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))
'''
model = ResNet50(include_top=True, input_tensor=None, input_shape=(img_channels, img_rows, img_cols))

# let's train the model using SGD + momentum (how original).

sgd = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')


lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(X_train, Y_train, batch_size=batch_size,
              nb_epoch=nb_epoch, validation_data= \
                        (X_test, Y_test), shuffle=True, callbacks=callbacks_list)
    model.save('Less_resnet50.h5')
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=True,  # apply ZCA whitening
        rotation_range=20,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.15,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.15,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen_test = ImageDataGenerator(
        featurewise_center=True,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=True,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=True,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.0,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.0,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)
    datagen_test.fit(X_test)

    # fit the model on the batches generated by datagen.flow()
    history = model.fit_generator(datagen.flow(X_train, Y_train , batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch, validation_data= \
                                  datagen_test.flow(X_test, Y_test, batch_size=batch_size),  \
                                  nb_val_samples=X_test.shape[0], nb_worker=1, \
                                  callbacks=callbacks_list)
    model.save('Less_resnet50.h5')
