import argparse
import os
import sys
import subprocess
import time
import tensorflow as tf
#import tensorflow.keras
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Softmax,LSTM,Activation, RNN,GRU, CuDNNGRU
from keras.layers import TimeDistributed, Bidirectional, Dropout, Lambda, Masking
from keras.constraints import max_norm
import keras.utils
import keras.backend as K
import numpy as np
import random
#from tf.keras.experimental import PeepholeLSTMCell
#from tf.keras.experimental import PeeholeLSTMCell
#import functools
#import CTCModel
import generator

os.environ['PYTHONHASHSEED']='0'
np.random.seed(1024)
random.seed(1024)

config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(allow_growth = True),
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

max_label_len=1024

def build_model(inputs, units, depth, n_labels, feat_dim, direction, use_softmax=True):

    #outputs = Masking(mask_value=0.0)(inputs)
    outputs=inputs
    for n in range (depth):
        if direction == 'bi':
            outputs=Bidirectional(CuDNNGRU(units,
                return_sequences=True))(outputs)
        else:
            outputs=CuDNNGRU(units,return_sequences=True)(outputs)
#        recurrent_activation='sigmoid',
        #outputs=Bidirectional(RNN(tf.compat.v1.keras.experimental.PeepholeLSTMCell(
        #                units, kernel_initializer='glorot_uniform',
        #                unit_forget_bias=True),
        #                return_sequences=True,
        #                ))(outputs)
#                                      dropout=0.1,
#                                      recurrent_dropout=0.1,
#                                      kernel_constraint=max_norm(2),
#                                      recurrent_constraint=max_norm(2),

    outputs = TimeDistributed(Dense(n_labels+1, name="timedist_dense"))(outputs)
    if use_softmax is True:
        outputs = Activation('softmax', name='softmax')(outputs)

#    model=CTCModel.CTCModel([inputs], [outputs], greedy=False)
    model.compile()

    return model

def named_logs(model, logs):
  result = {}
  for l in zip(model.metrics_names, logs):
    result[l[0]] = l[1]
  return result

def main():

    #print (keras.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='training data')
    parser.add_argument('--feat-dim', default=40, type=int, help='feats dim')
    '''
    n_labels = 32 librispeech
    n_labels = 49 tedlium_v1
    '''
    parser.add_argument('--n-labels', default=1024, type=int, required=True,
                        help='number of output labels')
    parser.add_argument('--batch-size', default=1, type=int, help='mini-batch size')
    parser.add_argument('--snapshot', type=str, default='./',
                        help='snapshot directory')
    parser.add_argument('--snapshot-prefix', type=str, default='eval',
                        help='snapshot file prefix')
    parser.add_argument('--weights', type=str, required=True, help='model weights')
    parser.add_argument('--prior', type=str, default=None, help='prior weights')
    parser.add_argument('--prior-scale', type=float, default=1.0, help='prior scaler')
    parser.add_argument('--units', type=int ,default=16, help='number of LSTM cells')
    parser.add_argument('--lstm-depth', type=int ,default=2,
                        help='number of LSTM/GRU layers')
    parser.add_argument('--direction', type=str, default='bi', help='RNN direction')
    parser.add_argument('--softmax', type=bool, default=True, help='use softmax layer')
    parser.add_argument('--align', type=bool, default=False, help='store alignment')
    args = parser.parse_args()

    inputs = Input(shape=(None, args.feat_dim))
    model = build_model(inputs, args.units, args.lstm_depth, args.n_labels,
        args.feat_dim, args.direction, args.use_softmax)
    model.load_weights(args.weights, by_name=True)

    prior = np.zeros((1, args.n_labels+1))
    if args.prior is not None:
        with h5py.File(arg.prior, 'r') as f:
            prior=f['prior'][()]
        prior=np.log(prior) # prior must be > 0
    prior=np.expand_dims(1, 0, prior)
    prior *= args.prior_scale;

    generator = generator.DataGenerator(args.data, None,
                            args.batch_size, args.feat_dim, args.n_labels)

    path=os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
    with h5py.File(path, 'w') as f:
        for bt in range(generator.__len__()):
            data, keys = training_generator.__getitem__(bt, return_keys=True)
            # data = [input_sequences, label_sequences, inputs_lengths, labels_length], keys
            # return softmax outputs
            predict = model.predict_on_batch(x=data[0])
            # shift for blank
            predict = np.roll(predict, 1, axis=2)
            if use_softmax is True:
                predict = np.log(predict)
            predict += prior

            for i, key in enumearte(keys):
                pr = np.squeeze(predict[i, :, :])
                f.create_dataset(key+'/likelihood', pr)

                if args.align is True:
                    lb = np.squeeze(data[1][i])
                    inlen=np.squeeze(data[2][i])
                    lblen=np.squeeze(data[3][i])

                    align = dynamic_programming(pr, lb, inlen, lblen)
                    f.create_dataset(key+'/align', align)

if __name__ == "__main__":
    main()
