import argparse
import os
import sys
import subprocess
import time
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Softmax,LSTM,Activation, CuDNNGRU, GRU, Reshape
from keras.layers import TimeDistributed, Bidirectional, Dropout, Lambda, Masking, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D
import keras.utils
import keras.backend as K
import numpy as np
import random
import tensorflow as tf
import ce_generator
import layer_normalization

def VGG2L_Strides(inputs, filters, feat_dim):

    outputs=Lambda(lambda x: tf.expand_dims(x, -1))(inputs)
    # first convs
    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)

    # half of the stride
    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=(2,1),
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)
    #outputs=MaxPooling2D(pool_size=2, strides=1, padding='same')(outputs)

    filters *= 2 # 128
    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)

    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=(2,1),
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)

    outputs = Reshape(target_shape=(-1, feat_dim*filters))(outputs)

    return outputs

def VGG2L_QuadStrides(inputs, filters, feat_dim):

    outputs=Lambda(lambda x: tf.expand_dims(x, -1))(inputs)
    # first convs
    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)

    # half of the stride
    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=(2,1),
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)
    #outputs=MaxPooling2D(pool_size=2, strides=2, padding='same')(outputs)

    filters *= 2
    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)

    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=(2,1),
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)
    #outputs=MaxPooling2D(pool_size=2, strides=2, padding='same')(outputs)

    outputs = Reshape(target_shape=(-1, feat_dim*filters))(outputs)

    return outputs

def VGG2L_Transpose(inputs, filters, units):
    outputs = Lambda(lambda x: tf.expand_dims(x, -1))(inputs)

    outputs = UpSampling2D(size=(2,1), data_format='channels_last')(outputs)

    outputs = Conv2D(filters=filters,
                     kernel_size=3, padding='same',
                     strides=(1,2),
                     data_format='channels_last',
                     kernel_initializer='glorot_uniform')(outputs)
    outputs = BatchNormalization(axis=-1)(outputs)
    outputs = Activation('relu')(outputs)

    outputs = Conv2D(filters=int(filters),
                 kernel_size=3, strides=(1,2),
                 padding='same',
                 data_format='channels_last',
                 kernel_initializer='glorot_uniform')(outputs)
    outputs = BatchNormalization(axis=-1)(outputs)
    outputs = Activation('relu')(outputs)

    outputs = Conv2D(filters=int(filters/2),
             kernel_size=3, strides=(1,2),
             padding='same',
             data_format='channels_last',
             kernel_initializer='glorot_uniform')(outputs)
    outputs = BatchNormalization(axis=-1)(outputs)
    outputs = Activation('relu')(outputs)

    # units/8, filters/2
    outputs = Reshape(target_shape=(-1, units)) (outputs)

    return outputs

def VGG2L_QuadTranspose(inputs, filters, units):
    outputs=Lambda(lambda x: tf.expand_dims(x, -1))(inputs)

    outputs = UpSampling2D(size=(2,1), data_format='channels_last')(outputs)

    outputs = Conv2D(filters=filters,
                     kernel_size=3, strides=(1,2),
                     padding='same',
                     data_format='channels_last',
                     kernel_initializer='glorot_uniform')(outputs)
    outputs = BatchNormalization(axis=-1)(outputs)
    outputs = Activation('relu')(outputs)

    outputs = UpSampling2D(size=(2,1), data_format='channels_last')(outputs)

    outputs = Conv2D(filters=filters,
                     kernel_size=3, strides=(1,2),
                     padding='same',
                     data_format='channels_last',
                     kernel_initializer='glorot_uniform')(outputs)
    outputs = BatchNormalization(axis=-1)(outputs)
    outputs = Activation('relu')(outputs)

    outputs = Conv2D(filters=int(filters/2),
                 kernel_size=3, strides=(1,2),
                 padding='same',
                 data_format='channels_last',
                 kernel_initializer='glorot_uniform')(outputs)
    outputs = BatchNormalization(axis=-1)(outputs)
    outputs = Activation('relu')(outputs)

    outputs = Reshape(target_shape=(-1, units))(outputs) # 64*8

    return outputs
