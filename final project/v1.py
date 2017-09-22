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
data_augmentation = False

# input image dimensions
img_rows, img_cols = 128, 128
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

model_resnet = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None)

input_resnet = Input(shape=(img_channels, img_rows, img_cols), name='image_input')

output_resnet = model_resnet(input_resnet)

x = Flatten(name='flatten')(output_resnet)
x = Dense(1024, init='he_normal', W_regularizer=l2(0.001), b_regularizer=l2(0.001), name='fc1')(x)
x = BatchNormalization(name='bn')(x)
x = Activation('relu', name='relu')(x)
x = Dropout(0.5)
x = Dense(nb_classes, name='pred')(x)
x = Activation('softmax',name='softmax')(x)

model = Model(input=input_resnet, output=x)

# let's train the model using SGD + momentum (how original).

sgd = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(X_train, Y_train, batch_size=batch_size,
              nb_epoch=nb_epoch, validation_data= \
                        (X_test, Y_test), shuffle=True, callbacks=callbacks_list)
    model.save('q1_v3.h5')
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
    model.save('q1_v3.h5')
print(history.history.keys())
# pickle.dump( history.history, open( "q1_v3.p", "wb" ) )
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('q1_v3_acc.png')
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('q1_v3_loss.png')
plt.clf()

test_err = [1 - x for x in history.history['val_acc']]
train_err = [1 - x for x in history.history['acc']]

# summarize history for error rate
plt.plot(train_err)
plt.plot(test_err)
plt.title('model error rate')
plt.ylabel('error rate')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('q1_v3_error.png')
plt.clf()
