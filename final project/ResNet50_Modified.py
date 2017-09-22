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
import matplotlib.pyplot as plt
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


# the data, shuffled and split between train and test sets
data = np.load('data224.npz')
X_test = data['arr_2']
Y_test = data['arr_3']



# input image dimensions
img_rows, img_cols = X_train.shape[2], X_train.shape[3]
# the CIFAR10 images are RGB
img_channels = 3

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train[:, 0], nb_classes)
Y_test = np_utils.to_categorical(Y_test[:, 0], nb_classes)


model = ResNet50(include_top=True, input_tensor=None, input_shape=(img_channels, img_rows, img_cols))


model = models.load_model('resnet50_2level.h5')
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

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('inceptionv3_acc.png')
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('inceptionv3_loss.png')
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
plt.savefig('inceptionv3_error.png')
plt.clf()