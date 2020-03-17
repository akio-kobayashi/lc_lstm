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

def lc_part_network(inputs1, inputs2, units, depth, n_labels, direction,
               dropout, init_filters, proc_frames, lstm=False):

    for n in range (depth):
        # forward, keep current states
        # statefule
        if lstm is False:
            forward_rnn_layer =  GRU(units, kernel_initializer='glorot_uniform',return_sequences=True, return_state=True,
                                        stateful=True, dropout=dropout, unroll=False)
        else:
            forward_rnn_layer = LSTM(units, kernel_initializer='glorot_uniform', return_sequences=True, return_state=True,
                                        stateful=True, use_forget_bias=True, dropout=dropout, unroll=False)
        # backward, not keep current states
        # do not preserve state values for backward pass
        if lstm is False:
            backward_rnn_layer = GRU(units, kernel_initializer='glorot_uniform', return_sequences=True, stateful=False,
                                    unroll=False, dropout=dropout, go_backwards=True)
        else:
            backward_rnn_layer = LSTM(units, kernel_initializer='glorot_uniform', return_sequences=True, stateful=False,
                                    unroll=False, usne_forget_bias=True, dropout=dropout, go_backwards=True)

        x1,h1 = forward_rnn_layer(inputs1)
        x2,h2 = forward_rnn_layer(inputs2)

        x = Concatenate()([x1, x2], axis=1)
        y = Concatenate()([inputs1, inputs2], axis=1)

        y = backward_rnn_layer(y)

        outputs = Concatenate(axis=-1)([x,y])
        outputs=layer_normalization.LayerNormalization()(outputs)

        inputs1 = Lambda(lambda x: x[:,0:proc_frames, :])(outputs)
        inputs2 = Lambda(lambda x: x[:,proc_frames:, :])(outputs)

        forward_rnn_layer.reset_states(h1)

    outputs = Concatenate()([inputs1, inputs2])
    return outputs
