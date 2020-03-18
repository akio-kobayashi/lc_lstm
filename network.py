from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Softmax,LSTM,Activation, CuDNNGRU, GRU, Reshape
from keras.layers import TimeDistributed, Bidirectional, Dropout, Lambda, Masking, Conv2D, CuDNNLSTM, Concatenate
import keras.utils
import keras.backend as K
import numpy as np
import tensorflow as tf
import layer_normalization

cudnn=True

def network(outputs, units, depth, n_labels, direction,
            dropout, init_filters, lstm=False):
    for n in range (depth):
        if direction == 'bi':
            if lstm is True:
                if cudnn is True:
                    outputs=Bidirectional(CuDNNLSTM(units,
                                                    return_sequences=True))(outputs)
                else:
                    outouts=Bidirectional(LSTM(units, kernel_initializer='glorot_uniform',
                                               return_sequences=True,
                                               use_forget_bias=True,
                                               dropout=dropout,
                                               unroll=False))(outputs)
            else:
                if cudnn is False:
                    outputs=Bidirectional(GRU(units,
                                              kernel_initializer='glorot_uniform',
                                              return_sequences=True,
                                              dropout=dropout,
                                              unroll=False))(outputs)
                else:
                    outputs=Bidirectional(CuDNNGRU(units,
                                                   return_sequences=True))(outputs)
        else:
            if lstm is True:
                if cudnn is True:
                    outputs = CuDNNLSTM(units, return_sequences=True)(outputs)
                else:
                    outputs=LSTM(units, kernel_initializer='glorot_uniform',
                                 return_sequences=True,
                                 use_forget_bias=True,
                                 dropout=dropout,
                                 unroll=False)(outputs)
            else:
                if cudnn is True:
                    outputs=CuDNNGRU(units,return_sequences=True)(outputs)
                else:
                    outputs=GRU(units,
                                kernel_initializer='glorot_uniform',
                                return_sequences=True,
                                dropout=dropout,
                                unroll=False)(outputs)
        outputs=layer_normalization.LayerNormalization()(outputs)

    return outputs

def lc_network(outputs, units, depth, n_labels, direction,
               dropout, init_filters, lstm=False):
    for n in range (depth):
        # forward, keep current states
        # statefule
        if lstm is False:
            x=GRU(units, kernel_initializer='glorot_uniform',
                  return_sequences=True,
                  stateful=True,
                  dropout=dropout,
                  unroll=False)(outputs)
        else:
            x=LSTM(units, kernel_initializer='glorot_uniform',
                   return_sequences=True,
                   stateful=True,
                   use_forget_bias=True,
                   dropout=dropout,
                   unroll=False)(outputs)
        # backward, not keep current states
        # do not preserve state values for backward pass
        if lstm is False:
            y=GRU(units, kernel_initializer='glorot_uniform',
                  return_sequences=True,
                  stateful=False,
                  unroll=False,
                  dropout=dropout,
                  go_backwards=True)(outputs)
        else:
            y=LSTM(units, kernel_initializer='glorot_uniform',
                   return_sequences=True,
                   stateful=False,
                   unroll=False,
                   usne_forget_bias=True,
                   dropout=dropout,
                   go_backwards=True)(outputs)

        outputs = Concatenate(axis=-1)([x,y])
        outputs=layer_normalization.LayerNormalization()(outputs)

    return outputs

def lc_part_network(inputs, units, depth, n_labels, direction,
               dropout, init_filters, proc_frames, lstm=False):

    forward_rnn_layers=[]
    backward_rnn_layers=[]

    z = inputs
    for n in range (depth):
        # forward, keep current states
        # statefule
        if lstm is False:
            forward_rnn_layer =  GRU(units, kernel_initializer='glorot_uniform',return_sequences=True, return_state=True,
                                        stateful=True, dropout=dropout, unroll=False)
        else:
            forward_rnn_layer = LSTM(units, kernel_initializer='glorot_uniform', return_sequences=True, return_state=True,
                                        stateful=True, use_forget_bias=True, dropout=dropout, unroll=False)
        forward_rnn_layers.append(forward_rnn_layer)
        # backward, not keep current states
        # do not preserve state values for backward pass
        if lstm is False:
            backward_rnn_layer = GRU(units, kernel_initializer='glorot_uniform', return_sequences=True, stateful=False,
                                    unroll=False, dropout=dropout, go_backwards=True)
        else:
            backward_rnn_layer = LSTM(units, kernel_initializer='glorot_uniform', return_sequences=True, stateful=False,
                                    unroll=False, usne_forget_bias=True, dropout=dropout, go_backwards=True)

        backward_rnn_layers.append(forward_rnn_layer)
        # concate
        x = forward_rnn_layer(z)
        y = backward_rnn_layer(z)

        z = Concatenate(axis=-1)([x,y])

        z = layer_normalization.LayerNormalization()(z)

    rnn_model = Model(inputs, z)

    trunc_z = Lambda(lambda x: x[:,0:proc_frames, :])(inputs)
    for n in range (depth):
        x = forward_rnn_layers[n](trunc_z)
        y = backward_rnn_layers[n](trunc_z)

        trun_z = Concatenate(axis=-1)([x,y])

    rnn_trunc_model = Model(inputs, trunc_z)
    rnn_trunc_model.trainable = False

    return rnn_model, rnn_trunc_model
