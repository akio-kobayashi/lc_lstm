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

    neg_outputs1=Lambda(lambda x: -1*x)(outputs)
    neg_outputs1=Activation('relu')(neg_outputs1)
    pos_outputs1=Activation('relu')(outputs)

    #filters*=2
    pos_outputs1=Conv2D(filters=filters,
                        kernel_size=3, padding='same',
                        strides=1,
                        data_format='channels_last',
                        kernel_initializer='glorot_uniform')(pos_outputs1)
    pos_outputs1=BatchNormalization(axis=-1)(pos_outputs1)
    neg_outputs2=Lambda(lambda x: -1*x)(pos_outputs1)
    neg_outputs2=Activation('relu')(neg_outputs2)
    pos_outputs2=Activation('relu')(pos_outputs1)
    #pos_outputs2=Reshape(target_shape=(-1, feat_dim*filters))(outputs)

    neg_outputs1=Conv2D(filters=filters,
                        kernel_size=3, padding='same',
                        strides=1,
                        data_format='channels_last',
                        kernel_initializer='glorot_uniform')(neg_outputs1)
    neg_outputs1=BatchNormalization(axis=-1)(neg_outputs1)
    neg_outputs3=Lambda(lambda x: -1*x)(neg_outputs1)
    neg_outputs3=Activation('relu')(neg_outputs3)
    pos_outputs3=Activation('relu')(neg_outputs1)
    #neg_outputs1=Reshape(target_shape=(-1, feat_dim*filters))(neg_outputs1)

    # pos_outputs2, neg_outputs2, pos_outputs3, neg_outputs3
    out_list=[]
    for out in [pos_outputs2, neg_outputs2, pos_outputs3, neg_outputs3]:
        out=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(out)
        out=BatchNormalization(axis=-1)(out)
        out=Activation('relu')(out)
        out_list.append(out)
    # 16*4=64 channels -> 16 channels
    outputs = Lambda(lambda x: tf.concat(x, axis=-1))(out_list)
    outputs=Conv2D(filters=filters,
                   kernel_size=3, padding='same',
                   strides=1,
                   data_format='channels_last',
                   kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)
    
    # 40 x 16 * 4 = 2560
    outputs = Reshape(target_shape=(-1, feat_dim * filters))(outputs)
    #outputs = Lambda(lambda x: tf.concat([x[0], x[1]], axis=-1))([outputs, extra_outputs])
    
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
