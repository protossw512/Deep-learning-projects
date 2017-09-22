from __future__ import print_function
from keras import models
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
# import matplotlib.pyplot as plt
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import AveragePooling2D
from keras.callbacks import LearningRateScheduler
import math
import numpy as np


# plt.ioff()


def step_decay(epoch):
    initial_lrate = 0.01
    drop = 0.15
    epochs_drop = 40.0
    lrate = initial_lrate * \
        math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


batch_size = 32
nb_classes = 29
nb_epoch = 200
data_augmentation = False

# input image dimensions
# img_rows, img_cols = 2048, 1
# the CIFAR10 images are RGB
# img_channels = 4028

# the data, shuffled and split between train and test sets
data = np.load('data299-tf.npz')
Y_train = data['arr_1']
Y_test = data['arr_3']
#X_train = np.load('bottleneck_features_train.npy')
All_train = np.load('xception.npz')
X_train = All_train['arr_0']
X_test = All_train['arr_1']

print(Y_train.shape)
print(Y_test.shape)
#print('X_train shape:', X_train.shape)
#print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

#img_rows, img_cols = X_train.shape[2], X_train.shape[3]
#img_channels = X_train.shape[1]

# convert class vectors to binary class matrices
#Y_train = np_utils.to_categorical(Y_train[:, 0], nb_classes)
#Y_test = np_utils.to_categorical(Y_test[:, 0], nb_classes)

#model = Sequential()
#
#model.add(AveragePooling2D(
#    pool_size=(8, 8),
#    input_shape=(img_channels, img_rows, img_cols))
#)
#model.add(BatchNormalization(input_shape=X_train.shape[1:]))
#model.add(Flatten())
#model.add(Dense(1024, init='he_normal'))
#model.add(BatchNormalization())
#model.add(Activation('relu'))
## model.add(Dense(1024, init='he_normal'))
## model.add(BatchNormalization())
## model.add(Activation('relu'))
#model.add(Dropout(0.3))
#model.add(Dense(nb_classes))
#model.add(Activation('softmax'))
#
## let's train the model using SGD + momentum (how original).
#
#sgd = SGD(lr=0.01, decay=0.0, momentum=0.9, nesterov=True)
#model.compile(
#    loss='categorical_crossentropy',
#    metrics=['accuracy', 'top_k_categorical_accuracy'],
#    optimizer=sgd
#)

#X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

#lrate = LearningRateScheduler(step_decay)
#callbacks_list = [lrate]

if not data_augmentation:
    print('Not using data augmentation.')
    model = models.load_model('res50.h5')
    hist = model.predict_classes(X_test, 32)
    yy = Y_test[:, 0].tolist()
    nn = Y_test.tolist()
    #yy = int(yy)
    res = [0 for x in xrange(29)]
    real = [0 for x in xrange(29)]
    for i in xrange(len(yy)):
        real[int(yy[i])] += 1
        if hist[i] == int(yy[i]):
            res[int(yy[i])] += 1
    print('\n')
    for i in xrange(len(res)):
        res[i] = float(res[i]) / float(real[i])
        name = ''
        for n in nn:
            if int(n[0]) == i:
                tmp = str(n[1])
                name = tmp[0:tmp.find('_')]
                break
    	print(name,' ',res[i]*100)
    #history = model.fit(
    #    X_train,
    #    Y_train,
    #    batch_size=batch_size,
    #    nb_epoch=nb_epoch,
    #    validation_data=(X_test, Y_test),
    #    shuffle=True,
    #    callbacks=callbacks_list
    #)
    #model.save('inceptionv3.h5')
else:
    print('Using real-time data augmentation.')


# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('q1_v3_acc.png')
# plt.clf()
# # summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('q1_v3_loss.png')
# plt.clf()

# test_err = [1 - x for x in history.history['val_acc']]
# train_err = [1 - x for x in history.history['acc']]

# # summarize history for error rate
# plt.plot(train_err)
# plt.plot(test_err)
# plt.title('model error rate')
# plt.ylabel('error rate')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('q1_v3_error.png')
# plt.clf()
