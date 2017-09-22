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
import numpy as np
from keras.utils import np_utils
from keras.applications.xception import Xception

nb_classes = 29

# input image dimensions
# img_rows, img_cols = 299, 299
# the CIFAR10 images are RGB
# img_channels = 3

# the data, shuffled and split between train and test sets
# (X_train, y_train), (X_test, y_test) = cifar10.load_data()
data = np.load('data299-tf.npz')
X_train = data['arr_0']
X_train = np.divide(X_train, 255.0)
X_train = np.subtract(X_train, 0.5)
X_train = np.multiply(X_train, 2.0)
Y_train = data['arr_1']
X_test = data['arr_2']
X_test = np.divide(X_test, 255.0)
X_test = np.subtract(X_test, 0.5)
X_test = np.multiply(X_test, 2.0)
Y_test = data['arr_3']

print(Y_train[:, 0].shape)
print(Y_test[:, 0].shape)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

img_rows, img_cols = X_train.shape[2], X_train.shape[3]
img_channels = X_train.shape[1]

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(Y_train[:, 0], nb_classes)
Y_test = np_utils.to_categorical(Y_test[:, 0], nb_classes)

model_inv3 = Xception(
    include_top=False,
    weights='imagenet',
    input_tensor=None,
    input_shape=(img_channels, img_rows, img_cols)
)



bottleneck_features_train = model_inv3.predict(X_train, 8)
bottleneck_features_test = model_inv3.predict(X_test, 8)
np.savez_compressed('xception.npz', bottleneck_features_train, bottleneck_features_test)
