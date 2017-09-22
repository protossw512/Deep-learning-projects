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
from keras.layers.pooling import GlobalAveragePooling2D
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.utils import np_utils
import matplotlib.pyplot as plt
import keras
from lsuv_init import LSUVinit
from keras.layers.normalization import BatchNormalization
from keras.callbacks import LearningRateScheduler
import math

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.02
	drop = 0.5
	epochs_drop = 50.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

batch_size = 32
nb_classes = 10
nb_epoch = 200
data_augmentation = True

# input image dimensions
img_rows, img_cols = 32, 32
# the CIFAR10 images are RGB
img_channels = 3

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(AveragePooling2D(pool_size=(2,2), input_shape=(img_channels, img_rows, img_cols)))
model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(48, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(48, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Convolution2D(48, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Convolution2D(80, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Convolution2D(80, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Convolution2D(80, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Convolution2D(80, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Convolution2D(80, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(GlobalAveragePooling2D())
model.add(Dropout(0.3))


# model.add(Flatten())
model.add(Dense(512, init='he_normal', W_regularizer=l2(0.05), b_regularizer=l2(0.05)))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.02, decay=0.0, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.03, beta_1=0.7, beta_2=0.8, epsilon=1e-08, decay=1e-6)
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

# model = LSUVinit(model,X_train[:batch_size,:,:,:])

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]
if not data_augmentation:
    print('Not using data augmentation.')
    history = model.fit(X_train, Y_train, batch_size=batch_size,
              nb_epoch=nb_epoch, show_accuracy=True,
                        validation_data=(X_test, Y_test), shuffle=True, callbacks=callbacks_list)
    model.save('q8.h5')
else:
    print('Using real-time data augmentation.')

    # this will do preprocessing and realtime data augmentation
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the model on the batches generated by datagen.flow()
    history = model.fit_generator(datagen.flow(X_train, Y_train , batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch, show_accuracy=True,
                        validation_data=(X_test, Y_test),
                                  nb_worker=1, callbacks=callbacks_list)
    model.save('q8.h5')
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('q8_acc.png')
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('q8_loss.png')
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
plt.savefig('q8_error.png')
plt.clf()
