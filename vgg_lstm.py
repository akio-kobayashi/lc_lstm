import argparse
import os
import sys
import subprocess
import time
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Softmax,LSTM,Activation, CuDNNGRU, GRU, Reshape, UpSampling2D
from keras.layers import TimeDistributed, Bidirectional, Dropout, Lambda, Masking, Conv2D, MaxPooling2D
import keras.utils
import keras.backend as K
import numpy as np
import random
import tensorflow as tf
import layer_normalization

def VGG3(inputs, mask, filters, feat_dim, dropout=0.0):

    outputs=Lambda(lambda x: tf.expand_dims(x, -1))(inputs)
    # first convs
    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)

    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)

    filters *= 2

    # original
    outputs1=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs1=BatchNormalization(axis=-1)(outputs1)
    outputs1=Activation('relu')(outputs1)

    outputs1=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs1)
    outputs1=BatchNormalization(axis=-1)(outputs1)
    outputs1=Activation('relu')(outputs1)
    outputs1 = Reshape(target_shape=(-1, feat_dim*filters))(outputs1)

    # half
    outputs2=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=(2,1),
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs2=BatchNormalization(axis=-1)(outputs2)
    outputs2=Activation('relu')(outputs2)

    outputs2=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs2)
    outputs2=BatchNormalization(axis=-1)(outputs2)
    outputs2=Activation('relu')(outputs2)
    outputs2 = Reshape(target_shape=(-1, feat_dim*filters))(outputs2)

    # quad
    outputs3=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=(2,1),
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs3=BatchNormalization(axis=-1)(outputs3)
    outputs3=Activation('relu')(outputs3)

    outputs3=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs3)
    outputs3=BatchNormalization(axis=-1)(outputs3)
    outputs3=Activation('relu')(outputs3)
    outputs3 = Reshape(target_shape=(-1, feat_dim*filters))(outputs3)

    mask2=Lambda(lambda x: tf.expand_dims(x, -1))(mask)
    mask2=MaxPooling2D(pool_size=(1, 1), strides=(2,1), padding='same', data_format='channels_last')(mask2)
    mask3=MaxPooling2D(pool_size=(1, 1), strides=(2,1), padding='same', data_format='channels_last')(mask2)

    return outputs1, outputs2, outputs3, mask2, mask3

def UpConvLSTM(inputs1, inputs2, inputs3, units, dropout=0.0):

    sunits = int(units/2)
    outputs2 = inputs2
    outputs3 = inputs3
    for n in range(2):
        output2f=GRU(sunits, kernel_initializer='glorot_uniform',
                    return_sequences=True,
                    stateful=True,
                    dropout=dropout,
                    unroll=False)(outputs2)

        output2b=GRU(sunits, kernel_initializer='glorot_uniform',
                    return_sequences=True,
                    stateful=False,
                    unroll=False,
                    dropout=dropout,
                    go_backwards=True)(outputs2)
        outputs2 = Concatenate(axis=-1)([outputs2f,outputs2b])
        outputs2 = layer_normalization.LayerNormalization()(outputs2)

        output3f=GRU(sunits, kernel_initializer='glorot_uniform',
                    return_sequences=True,
                    stateful=True,
                    dropout=dropout,
                    unroll=False)(outputs3)

        output3b=GRU(sunits, kernel_initializer='glorot_uniform',
                    return_sequences=True,
                    stateful=False,
                    unroll=False,
                    dropout=dropout,
                    go_backwards=True)(outputs3)
        outputs3 = Concatenate(axis=-1)([outputs3f,outputs3b])
        outputs3 = layer_normalization.LayerNormalization()(outputs3)

    outputs3 = Lambda(lambda x: tf.expand_dims(x, -1))(outputs3)
    outputs3 = UpSampling2D(size=(2,1), data_format='channels_last')(outputs3)
    outputs3 = Reshape(target_shape=(-1, sunits*2)) (outputs3)

    outputs2 = Concatenate(axis=-1)([outputs2,outputs3]) # sunits * 4

    for n in range(2):
        output2f=GRU(sunits, kernel_initializer='glorot_uniform',
                    return_sequences=True,
                    stateful=True,
                    dropout=dropout,
                    unroll=False)(outputs2)

        output2b=GRU(sunits, kernel_initializer='glorot_uniform',
                    return_sequences=True,
                    stateful=False,
                    unroll=False,
                    dropout=dropout,
                    go_backwards=True)(outputs2)
        outputs2 = Concatenate(axis=-1)([outputs2f,outputs2b])
        outputs2 = layer_normalization.LayerNormalization()(outputs2)

    outputs2 = Lambda(lambda x: tf.expand_dims(x, -1))(outputs2)
    outputs2 = UpSampling2D(size=(2,1), data_format='channels_last')(outputs2)
    outputs2 = Reshape(target_shape=(-1, sunits*2)) (outputs2)

    outputs1 = inputs1
    for n in range(2):
        output1f=GRU(units, kernel_initializer='glorot_uniform',
                    return_sequences=True,
                    stateful=True,
                    dropout=dropout,
                    unroll=False)(outputs1)

        output1b=GRU(units, kernel_initializer='glorot_uniform',
                    return_sequences=True,
                    stateful=False,
                    unroll=False,
                    dropout=dropout,
                    go_backwards=True)(inputs1)
        outputs1 = Concatenate(axis=-1)([outputs1f,outputs1b])
        outputs1 = layer_normalization.LayerNormalization()(outputs1)

    outputs1 = Concatenate(axis=-1)([outputs1,outputs2]) # units*2 + sunits*2

    for n in range(2):
        output1f=GRU(units, kernel_initializer='glorot_uniform',
                    return_sequences=True,
                    stateful=True,
                    dropout=dropout,
                    unroll=False)(outputs1)

        output1b=GRU(units, kernel_initializer='glorot_uniform',
                    return_sequences=True,
                    stateful=False,
                    unroll=False,
                    dropout=dropout,
                    go_backwards=True)(inputs1)
        outputs1 = Concatenate(axis=-1)([outputs1f,outputs1b])
        outputs1 = layer_normalization.LayerNormalization()(outputs1)

    return outputs1
