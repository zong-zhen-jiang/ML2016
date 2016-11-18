from __future__ import print_function
from keras import backend as K
from keras import objectives
from keras.layers import Dense, Dropout, Input, Flatten, Lambda, Reshape
from keras.layers import Convolution2D, Deconvolution2D, MaxPooling2D
from keras.models import load_model, Sequential
from keras.models import Model
from keras.optimizers import Adagrad, Adam, SGD
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from sklearn.ensemble import RandomForestClassifier
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
    print ('Loading label data...')
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
label_x, label_y_1 = load_label_data(data_dir)
# plt.imshow(label_x[0])
# plt.show()
label_x = label_x.astype('float32') / 255
# (5000, 10)
label_y = np_utils.to_categorical(label_y_1, 10)

# (45000, 32, 32, 3)
unlabel_x = load_unlabel_data(data_dir)
# plt.imshow(unlabel_x[0])
# plt.show()
unlabel_x = unlabel_x.astype('float32') / 255

#
train_x = np.concatenate((label_x, unlabel_x))


###############################################################################
# Model definition: Variational AutoEncoder (VAE)
###############################################################################

batch_size = 2500
epsilon_std = 1.0
img_cols = 32
img_rows = 32
intermediate_dim = 1024
latent_dim = 256

inpu = Input(batch_shape=(batch_size,) + (32, 32, 3))
conv_1 = Convolution2D(3, 2, 2, border_mode='same', activation='relu')(inpu)
conv_2 = Convolution2D(
        64, 2, 2,
        border_mode='same',
        activation='relu',
        subsample=(2, 2))(conv_1)
conv_3 = Convolution2D(
        64, 3, 3,
        border_mode='same',
        activation='relu',
        subsample=(1, 1))(conv_2)
conv_4 = Convolution2D(
        64, 3, 3,
        border_mode='same',
        activation='relu',
        subsample=(1, 1))(conv_3)
flat = Flatten()(conv_4)
hidden = Dense(intermediate_dim, activation='relu')(flat)

z_mean = Dense(latent_dim)(hidden)
z_log_var = Dense(latent_dim)(hidden)


def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(
            shape=(batch_size, latent_dim), mean=0.0, std=epsilon_std)
    return z_mean + K.exp(z_log_var) * epsilon

# note that "output_shape" isn't necessary with the TensorFlow backend
# so you could write `Lambda(sampling)([z_mean, z_log_var])`
z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

# we instantiate these layers separately so as to reuse them later
decoder_hid = Dense(intermediate_dim, activation='relu')
decoder_upsample = Dense(64 * 16 * 16, activation='relu')

#
output_shape = (batch_size, 16, 16, 64)
decoder_reshape = Reshape(output_shape[1:])
decoder_deconv_1 = Deconvolution2D(
        64, 3, 3,
        output_shape,
        border_mode='same',
        subsample=(1, 1),
        activation='relu')
decoder_deconv_2 = Deconvolution2D(
        64, 3, 3,
        output_shape,
        border_mode='same',
        subsample=(1, 1),
        activation='relu')

#
output_shape = (batch_size, 33, 33, 64)
decoder_deconv_3_upsamp = Deconvolution2D(
        64, 2, 2,
        output_shape,
        border_mode='valid',
        subsample=(2, 2),
        activation='relu')
decoder_mean_squash = Convolution2D(
        3, 2, 2,
        border_mode='valid',
        activation='sigmoid')

hid_decoded = decoder_hid(z)
up_decoded = decoder_upsample(hid_decoded)
reshape_decoded = decoder_reshape(up_decoded)
deconv_1_decoded = decoder_deconv_1(reshape_decoded)
deconv_2_decoded = decoder_deconv_2(deconv_1_decoded)
x_decoded_relu = decoder_deconv_3_upsamp(deconv_2_decoded)
x_decoded_mean_squash = decoder_mean_squash(x_decoded_relu)

def vae_loss(inpu, x_decoded_mean):
    # NOTE: binary_crossentropy expects a batch_size by dim
    # for x and x_decoded_mean, so we MUST flatten these!
    inpu = K.flatten(inpu)
    x_decoded_mean = K.flatten(x_decoded_mean)
    xent_loss = img_rows * img_cols * \
            objectives.binary_crossentropy(inpu, x_decoded_mean)
    kl_loss = - 0.5 * \
            K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss

# Turns out that Adam fucks!
# adagrad = Adagrad(lr=0.01, epsilon=1e-08, decay=0.0)
adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
# sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
vae = Model(inpu, x_decoded_mean_squash)
vae.compile(optimizer=adam, loss=vae_loss)
# vae.summary()


###############################################################################
# Training the autoencoder
###############################################################################

vae.fit(train_x, train_x,
        shuffle=True,
        nb_epoch=100,
        batch_size=batch_size,
        validation_data=None)

# Save the 1st model (VAE encoder part).
encoder = Model(inpu, z_mean)
encoder.save(out_model + '_1')


###############################################################################
# Training the classifier
###############################################################################

# (5000, latent_dim)
label_x_encoded = encoder.predict(label_x, batch_size=batch_size, verbose=1)

#
print('Training the classifier...')
rf = RandomForestClassifier(n_estimators=300, max_features=128)
rf.fit(label_x_encoded, label_y_1.reshape(5000,))

# Save the 2nd model (classifier).
cPickle.dump(rf, open(out_model + '_2', 'wb'), cPickle.HIGHEST_PROTOCOL)
