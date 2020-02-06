import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Softmax, LSTM, Activation, RNN, GRU, CuDNNGRU
from keras.layers import TimeDistributed, Bidirectional, Dropout, Lambda, Masking
from keras.layers import Conv2D, Reshape, Concatenate
from keras.constraints import max_norm
import keras.utils
import keras.backend as K
import numpy as np
import random
import time
import CTCModel
import layer_normalization

def build_bipolar_model(inputs, units, depth, n_labels, feat_dim, init_lr, direction,
                dropout, init_filters, optim):

    #outputs = Masking(mask_value=0.0)(inputs)
    outputs=inputs

    # add channel dim
    outputs=Lambda(lambda x: tf.expand_dims(x, -1))(outputs)

    filters=init_filters
    outputs=Conv2D(filters=filters,
        kernel_size=3, padding='same',
        strides=1,
        data_format='channels_last',
        kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    #extra_outputs = -1*outputs
    extra_outputs = Lambda(lambda x: -1*x)(outputs)
    extra_outputs = Activation('relu')(extra_outputs)
    outputs=Activation('relu')(outputs)

    outputs=Conv2D(filters=filters,
        kernel_size=3, padding='same',
        strides=1,
        data_format='channels_last',
        kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)
    outputs = Reshape(target_shape=(-1, feat_dim*filters))(outputs)

    extra_outputs=Conv2D(filters=filters,
            kernel_size=3, padding='same',
            strides=1,
            data_format='channels_last',
            kernel_initializer='glorot_uniform')(extra_outputs)
    extra_outputs=BatchNormalization(axis=-1)(extra_outputs)
    extra_outputs=Activation('relu')(extra_outputs)
    extra_outputs = Reshape(target_shape=(-1, feat_dim*filters))(extra_outputs)

    #outputs = Concatenate(axis=-1)([outputs, extra_outputs])
    outputs = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([outputs, extra_outputs])
    
    for n in range (depth):
        if direction == 'bi':
            outputs=Bidirectional(CuDNNGRU(units,
                return_sequences=True))(outputs)
        else:
            outputs=CuDNNGRU(units,return_sequences=True)(outputs)
        outputs=layer_normalization.LayerNormalization()(outputs)

    outputs = TimeDistributed(Dense(n_labels+1, name="timedist_dense"))(outputs)
    outputs = Activation('softmax', name='softmax')(outputs)

    model=CTCModel.CTCModel([inputs], [outputs], greedy=True)
    if optim == 'adam':
        model.compile(keras.optimizers.Adam(lr=init_lr, clipnorm=50.))
    else:
        model.compile(keras.optimizers.Adadelta())

    return model
