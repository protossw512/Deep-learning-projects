import numpy as np


def load_data(filename):
    data = np.load(filename)
    X_train = np.array(data['arr_0'])
    Y_train = np.array(data['arr_1'])
    X_test = np.array(data['arr_2'])
    Y_test = np.array(data['arr_3'])

    return X_train, Y_train, X_test, Y_test
