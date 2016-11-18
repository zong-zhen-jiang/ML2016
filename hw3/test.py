from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.optimizers import Adagrad, Adam, SGD
from keras.utils import np_utils
from time import sleep

import cv2
import cPickle
# Keira is my wife >////////<
import keras as mywife
import matplotlib.pyplot as plt
import numpy as np
import sys


# TAs are so so so handsome :)
mywife.backend.set_image_dim_ordering('tf')


###############################################################################
# Input arguments
###############################################################################

data_dir = sys.argv[1]
in_model = sys.argv[2]
out_csv = sys.argv[3]
print('(data_dir, in_model, out_csv) = (\'%s\', \'%s\', \'%s\')' % \
        (data_dir, in_model, out_csv))


###############################################################################
# Loading test data
###############################################################################

# Load test data.
def load_test_data(data_dir):
    print('Loading test data...')
    test_data = cPickle.load(open(data_dir + '/test.p', 'rb'))
    test_id = np.array(test_data['ID'])

    test_x = [] # (10000, 32, 32, 3)
    for i in xrange(10000):
        # (3072) -> (3, 32, 32) -> (32, 32, 3)
        img = np.array(test_data['data'][i], dtype='uint8')
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)
        # BRG -> RGB
        c0, c1, c2 = cv2.split(img)
        img = cv2.merge((c1, c2, c0))
        #
        test_x.append(img)
    #
    test_x = np.array(test_x)
    return test_x, test_id


# x: (10000, 32, 32, 3), id: (10000,)
test_x, test_id = load_test_data(data_dir)
# This is fucking important!
test_x = test_x.astype('float32') / 255


###############################################################################
# Load the model to predict test data and save the result to CSV.
###############################################################################

model = load_model(in_model)

# (10000, 10)
test_y = model.predict_classes(test_x, verbose=1)

# Save to CSV.
np.savetxt(out_csv, np.column_stack((test_id, test_y)),
        header='ID,class', comments='', fmt='%s', delimiter=',')
