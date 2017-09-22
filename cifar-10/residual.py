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
from keras.layers import merge, Input
from keras.models import Sequential, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, ZeroPadding2D, MaxPooling2D, AveragePooling2D
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

# learning rate schedule
def step_decay(epoch):
	initial_lrate = 0.01
	drop = 0.1
	epochs_drop = 50.0
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate


def pop_layer(Model):
    if not Model.outputs:
        raise Exception('Sequential Model cannot be popped: Model is empty.')

    Model.layers.pop()
    if not Model.layers:
        Model.outputs = []
        Model.inbound_nodes = []
        Model.outbound_nodes = []
    else:
        Model.layers[-1].outbound_nodes = []
        Model.outputs = [Model.layers[-1].output]
    Model.built = False

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity_block is the block that has no conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    if keras.backend.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, name=conv_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size,
                      border_mode='same', name=conv_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)

    x = merge([x, input_tensor], mode='sum')
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut

    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the nb_filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names

    Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
    And the shortcut should have subsample=(2,2) as well
    """
    nb_filter1, nb_filter2, nb_filter3 = filters
    if keras.backend.image_dim_ordering() == 'tf':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Convolution2D(nb_filter1, 1, 1, subsample=strides,
                      name=conv_name_base + '2a')(input_tensor)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter2, kernel_size, kernel_size, border_mode='same',
                      name=conv_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Convolution2D(nb_filter3, 1, 1, name=conv_name_base + '2c')(x)

    shortcut = Convolution2D(nb_filter3, 1, 1, subsample=strides,
                             name=conv_name_base + '1')(input_tensor)

    x = merge([x, shortcut], mode='sum')
    x = Activation('relu')(x)
    return x

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

if keras.backend.image_dim_ordering() == 'tf':
    bn_axis = 3
else:
    bn_axis = 1
img_input = Input(shape=(img_channels, img_rows, img_cols))

x = ZeroPadding2D((3, 3))(img_input)
x = Convolution2D(32, 7, 7, subsample=(2, 2), name='conv1')(x)
x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
x = Activation('relu')(x)
x = MaxPooling2D((3, 3), strides=(2, 2))(x)

x = conv_block(x, 3, [32, 64, 128], stage=2, block='a', strides=(1, 1))
x = identity_block(x, 3, [32, 64, 128], stage=2, block='b')
x = identity_block(x, 3, [32, 64, 128], stage=2, block='c')

x = conv_block(x, 3, [64, 64, 128], stage=3, block='a')
x = identity_block(x, 3, [64, 64, 128], stage=3, block='b')
x = identity_block(x, 3, [64, 64, 128], stage=3, block='c')
x = identity_block(x, 3, [64, 64, 128], stage=3, block='d')

x = conv_block(x, 3, [128, 128, 256], stage=4, block='a')
x = identity_block(x, 3, [128, 128, 256], stage=4, block='b')
x = identity_block(x, 3, [128, 128, 256], stage=4, block='c')
x = identity_block(x, 3, [128, 128, 256], stage=4, block='d')
x = identity_block(x, 3, [128, 128, 256], stage=4, block='e')
x = identity_block(x, 3, [128, 128, 256], stage=4, block='f')

x = conv_block(x, 3, [256, 256, 512], stage=5, block='a')
x = identity_block(x, 3, [256, 256, 512], stage=5, block='b')
x = identity_block(x, 3, [256, 256, 512], stage=5, block='c')

x = AveragePooling2D((3, 3), name='avg_pool')(x)

x = Flatten()(x)
x = Dense(512, init='he_normal', W_regularizer=l2(0.001), b_regularizer=l2(0.001))(x)
x = Activation('relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(nb_classes)(x)
x = Activation('softmax')

Model = Model(img_input, x, name='resnet_test')
# let's train the Model using SGD + momentum (how original).
sgd = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam(lr=0.005, beta_1=0.9, beta_2=0.999, epsilon=1e-06, decay=0.0001)
Model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=sgd)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

lrate = LearningRateScheduler(step_decay)
callbacks_list = [lrate]

if not data_augmentation:
    print('Not using data augmentation.')
    history = Model.fit(X_train, Y_train, batch_size=batch_size,
              nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test), shuffle=True, callbacks=callbacks_list)
    Model.save('q4_v3-5.h5')
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
            horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    # fit the Model on the batches generated by datagen.flow()
    history = Model.fit_generator(datagen.flow(X_train, Y_train , batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test),
                                  nb_worker=1, callbacks=callbacks_list)
    Model.save('q4_v3-5.h5')
# print(history.history.keys())
# pickle.dump( history.history, open( "q4_v3-5.p", "wb" ) )
# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('Model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('q4_v3-5_acc.png')
# plt.clf()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('q4_v3-5_loss.png')
# plt.clf()

# test_err = [1 - x for x in history.history['val_acc']]
# train_err = [1 - x for x in history.history['acc']]

# # summarize history for error rate
# plt.plot(train_err)
# plt.plot(test_err)
# plt.title('Model error rate')
# plt.ylabel('error rate')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('q4_v3-5_error.png')
# plt.clf()
