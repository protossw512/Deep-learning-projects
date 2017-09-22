from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import re

idx = 0
cnt = 0
test = 0
train = 0
train_x = []
train_y = []
test_x = []
test_y = []

root = 'image224/'

rate = 0.15

np.random.seed(128)

for name in os.listdir(root):
    print "Start load ", name
    if re.match('^[A-Z]', name) and os.path.isdir(root + name):
        for r, ds, fs in os.walk(root + name):
            max_val = -1
            for f in fs:
                end = f.find('specimen')\
                    if f.find('specimen') != -1\
                    else f.find('background')
                tmp_val = int(f[len(name) + 1:end - 1])
                max_val = max_val if max_val >= tmp_val else tmp_val
            print "max_val: ", max_val
            arr = np.arange(1, max_val + 1)
            np.random.shuffle(arr)
            print "arr.size: ", arr.size
            for i in xrange(arr.size):
                for f in fs:
                    end = f.find('specimen')\
                        if f.find('specimen') != -1\
                        else f.find('background')
                    tmp_val = int(f[len(name) + 1:end - 1])
		    if tmp_val == arr[i]:
			cnt += 1
                        img = img_to_array(
                            load_img(
                                os.path.join(r, f),
                                grayscale=False
                            )
                        )
                        if i <= int(arr.size * rate):
                            test += 1
			    test_x.append(img)
                            test_y.append([idx, f])
                        else:
			    train += 1
                            train_x.append(img)
                            train_y.append([idx, f])
	print "cnt ", cnt, ", test ", test, ", train ", train
        print name, " finished loading"
        idx += 1

test_x = np.asarray(test_x)
print "test_x shape ", test_x.shape
test_y = np.asarray(test_y)
print "test_y shape ", test_y.shape
train_x = np.asarray(train_x)
print "train_x shape ", train_x.shape
train_y = np.asarray(train_y)
print "train_y shape ", train_y.shape

# print "Begin save test_x array to file test_x"
# np.save('test_x', test_x)
# print "Finished"
# print "Begin save test_y array to file text_y"
# np.save('test_y', test_y)
# print "Finished"
# print "Begin save train_x array to file train_x"
# np.save('train_x', train_x)
# print "Finished"
# print "Begin save train_y array to file train_y"
# np.save('train_y', train_y)
# print "Finished"

np.savez_compressed('data.npz', train_x, train_y, test_x, test_y)
