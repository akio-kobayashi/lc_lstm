import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Input, BatchNormalization, Softmax, LSTM, Activation, RNN, GRU, CuDNNGRU
from keras.layers import TimeDistributed, Bidirectional, Dropout, Lambda, Masking
from keras.layers import Conv2D, Reshape, Concatenate, MaxPooling2D
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

    # first convs
    filters=init_filters
    outputs=Conv2D(filters=filters,
        kernel_size=3, padding='same',
        strides=1,
        data_format='channels_last',
        kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)

    neg_outputs1=Lambda(lambda x: -1*x)(outputs)
    neg_outputs1=Activation('relu')(neg_outputs1)
    pos_outputs1=Activation('relu')(outputs)

    pos_outputs1=Conv2D(filters=filters,
                        kernel_size=3, padding='same',
                        strides=1,
                        data_format='channels_last',
                        kernel_initializer='glorot_uniform')(pos_outputs1)
    pos_outputs1=BatchNormalization(axis=-1)(pos_outputs1)
    neg_outputs2=Lambda(lambda x: -1*x)(pos_outputs1)
    neg_outputs2=Activation('relu')(neg_outputs2)
    pos_outputs2=Activation('relu')(pos_outputs1)

    #pos_outputs1=MaxPooling2D(pool_size=2, strides=1, padding='same')(pos_outputs1)
    #pos_outputs2=MaxPooling2D(pool_size=2, strides=1, padding='same')(pos_outputs2)
    #neg_outputs1=MaxPooling2D(pool_size=2, strides=1, padding='same')(neg_outputs1)
    #neg_outputs2=MaxPooling2D(pool_size=2, strides=1, padding='same')(neg_outputs2)

    conv_list=[pos_outputs1, pos_outputs2, neg_outputs1, neg_outputs2]
    conv_2nd=[]

    # second convs
    filters*=2
    for conv in conv_list:
        conv=Conv2D(filters=filters,
                    kernel_size=3, padding='same',
                    strides=1,
                    data_format='channels_last',
                    kernel_initializer='glorot_uniform')(conv)
        conv=BatchNormalization(axis=-1)(conv)
        conv=Activation('relu')(conv)
        conv=Conv2D(filters=filters,
                    kernel_size=3, padding='same',
                    strides=1,
                    data_format='channels_last',
                    kernel_initializer='glorot_uniform')(conv)
        conv=BatchNormalization(axis=-1)(conv)
        conv=Activation('relu')(conv)
        #conv=MaxPooling2D(pool_size=2, strides=1, padding='same')(conv)
        conv_2nd.append(conv)

    outputs = Lambda(lambda x: tf.concat(x, axis=-1))(conv_2nd)

    # 40 x 512 = 20480
    outputs = Reshape(target_shape=(-1, feat_dim * 4* filters))(outputs)
    
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
