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
out_model = sys.argv[2]
print('(data_dir, out_model) = (\'%s\', \'%s\')' % (data_dir, out_model))


###############################################################################
# Functions definition
###############################################################################

def load_label_data(data_dir):
    print('Loading label data...')
    # Input shape: (10, 500, 3072)
    label_data = cPickle.load(open(data_dir + '/all_label.p', 'rb'))
    label_data = np.array(label_data, dtype='uint8')

    # Split input to image data (x) and label (y).
    label_x = [] # (5000, 32, 32, 3)
    label_y = [] # (5000, 1)
    for c in xrange(10):
        for i in xrange(500):
            # (3072) -> (3, 32, 32) -> (32, 32, 3)
            img = label_data[c][i]
            img = img.reshape(3, 32, 32)
            img = img.transpose(1, 2, 0)
            # BRG -> RGB
            c0, c1, c2 = cv2.split(img)
            img = cv2.merge((c1, c2, c0))
            #
            label_x.append(img)
            label_y.append([c])
    # 
    label_x = np.array(label_x)
    label_y = np.array(label_y)
    return label_x, label_y

def load_unlabel_data(data_dir):
    print('Loading unlabel data...')
    unlabel_data = cPickle.load(open(data_dir + '/all_unlabel.p', 'rb'))
    unlabel_data = np.array(unlabel_data) # (45000, 3072)

    unlabel_x = [] # (45000, 32, 32, 3)
    for i in xrange(45000):
        # (3072) -> (3, 32, 32) -> (32, 32, 3)
        img = np.array(unlabel_data[i], dtype='uint8')
        img = img.reshape(3, 32, 32)
        img = img.transpose(1, 2, 0)
        # BRG -> RGB
        c0, c1, c2 = cv2.split(img)
        img = cv2.merge((c1, c2, c0))
        #
        unlabel_x.append(img)
    #
    unlabel_x = np.array(unlabel_x)
    return unlabel_x


###############################################################################
# Loading (label/unlabel) data
###############################################################################

# x: (5000, 32, 32, 3), y: (5000, 1)
label_x, label_y = load_label_data(data_dir)
# plt.imshow(label_x[0])
# plt.show()
label_x = label_x.astype('float32') / 255
# (5000, 10)
label_y = np_utils.to_categorical(label_y, 10)

# (45000, 32, 32, 3)
unlabel_x = load_unlabel_data(data_dir)
# plt.imshow(unlabel_x[0])
# plt.show()
unlabel_x = unlabel_x.astype('float32') / 255


###############################################################################
# Model definition
###############################################################################

model = Sequential()

model.add(Convolution2D(
        64, 3, 3, border_mode='same', input_shape=label_x.shape[1:]))
model.add(LeakyReLU(alpha=0.3))
model.add(Convolution2D(64, 3, 3))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(32, 3, 3, border_mode='same'))
model.add(LeakyReLU(alpha=0.3))
model.add(Convolution2D(32, 3, 3))
model.add(LeakyReLU(alpha=0.3))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))
model.add(Dense(512))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.5))
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.25))
model.add(Dense(256))
model.add(LeakyReLU(alpha=0.3))
model.add(Dropout(0.25))

model.add(Dense(10))
model.add(Activation('softmax'))

# Turns out that Adam fucks!
# adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(
        loss='categorical_crossentropy',
        optimizer=adam,
        metrics=['accuracy'])


###############################################################################
# Supervised training (label data only)
###############################################################################

# Data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=10,
        width_shift_range=0.10,
        height_shift_range=0.10,
        horizontal_flip=True,
        vertical_flip=False,
        zoom_range=0.15)
datagen.fit(label_x)

model.fit_generator(
        datagen.flow(label_x, label_y, batch_size=500),
        samples_per_epoch=label_x.shape[0],
        nb_epoch=4000,
        validation_data=None)

#
model.save(out_model + '_sup')

###############################################################################
# Semi-supervised training
###############################################################################

fuck = 5000
stop_unlabel_size = 20000
while (len(unlabel_x) > stop_unlabel_size):
    print('\n\nPredicting unlabel set...')
    # (len(unlabel_x), 10)
    unlabel_y_prob = model.predict(unlabel_x, verbose=1)

    # (len(unlabel_x), 3)
    tmp = []
    for i in xrange(len(unlabel_x)):
        clas = np.argmax(unlabel_y_prob[i])
        max_p = unlabel_y_prob[i][clas]
        tmp.append([i, max_p, clas])
    #
    tmp = np.array(tmp)
    tmp = tmp[tmp[:, 1].argsort()]
    tmp = tmp[::-1]

    # Extract unlabel data with high predicted probability.
    label_x_2 = unlabel_x[tmp[:fuck, 0].astype('int')]
    label_y_2 = tmp[:fuck, 2].astype('int').reshape(fuck, 1)
    label_y_2 = np_utils.to_categorical(label_y_2, 10)

    # Delete the extracted data from unlabel set.
    unlabel_x = np.delete(unlabel_x, tmp[:fuck, 0].astype('int'), axis=0)
    print('\n####################################################')
    print('# Size of unlabel set: %d' % (len(unlabel_x)))

    # Cluster the extracted data into label set.
    label_x = np.concatenate((label_x, label_x_2))
    label_y = np.concatenate((label_y, label_y_2))
    print('# Size of label set: %d' % (len(label_x)))
    print('# Stop when unlabel size <= %d' % (stop_unlabel_size))
    print('####################################################')
    datagen.fit(label_x)
    model.fit_generator(
            datagen.flow(label_x, label_y, batch_size=1000),
            samples_per_epoch=label_x.shape[0],
            nb_epoch=len(unlabel_x)/200,
            validation_data=None)

# Save the trained model.
model.save(out_model)
