import argparse
import os
import sys
import subprocess
import time
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Softmax,LSTM,Activation, RNN,GRU, CuDNNGRU
from keras.layers import TimeDistributed, Bidirectional, Dropout, Lambda, Masking, Conv2D, Reshape
from keras.constraints import max_norm
import keras.utils
import keras.backend as K
import numpy as np
import random
import h5py
import generator
import dynamic_programming
import layer_normalization
import vgg2l

os.environ['PYTHONHASHSEED']='0'
np.random.seed(1024)
random.seed(1024)
np.set_printoptions(threshold=np.inf)

config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(allow_growth = True),
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(config=config)
K.set_session(sess)

max_label_len=1024

def build_model(inputs, units, depth, n_labels, feat_dim,
                direction, init_filters):

    outputs = Masking(mask_value=0.0)(inputs)
    outputs = vgg2l.VGG2L(inputs, init_filters, feat_dim)
    
    for n in range (depth):
        if direction == 'bi':
            outputs=Bidirectional(CuDNNGRU(units,
                return_sequences=True))(outputs)
        else:
            outputs=CuDNNGRU(units,return_sequences=True)(outputs)
        outputs=layer_normalization.LayerNormalization()(outputs)

    outputs = TimeDistributed(Dense(n_labels+1))(outputs)
    outputs = Activation('softmax')(outputs)
    model=Model(inputs, outputs)

    return model

def named_logs(model, logs):
  result = {}
  for l in zip(model.metrics_names, logs):
    result[l[0]] = l[1]
  return result

def main():

    #print (keras.__version__)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data')
    parser.add_argument('--key-file', type=str, help='keys')
    parser.add_argument('--feat-dim', default=40, type=int, help='feats dim')
    '''
    n_labels = 32 librispeech
    n_labels = 49 tedlium_v1
    '''
    parser.add_argument('--n-labels', default=1024, type=int, required=True,
                        help='number of output labels')
    parser.add_argument('--batch-size', default=16, type=int, help='mini-batch size')
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
    #parser.add_argument('--softmax', type=bool, default=True, help='use softmax layer')
    #parser.add_argument('--align', type=bool, default=False, help='store alignment')
    #parser.add_argument('--layer-norm', type=bool, default=False, help='layer normalization')
    #parser.add_argument('--vgg', type=bool, default=False, help='use vgg-like layers')
    parser.add_argument('--filters', type=int, default=16, help='number of filters for CNNs')
    args = parser.parse_args()

    inputs = Input(shape=(None, args.feat_dim))
    model = build_model(inputs, args.units, args.lstm_depth, args.n_labels,
                        args.feat_dim, args.direction,
                        args.filters)
    model.load_weights(args.weights, by_name=True)

    prior = np.zeros((1, args.n_labels+1))
    if args.prior is not None:
        with h5py.File(args.prior, 'r') as f:
            prior=f['counts'][()]
        # reshape because prior is a column vector
        prior=prior.reshape((1, prior.shape[0]))
        prior=np.log(prior) # prior has been normalized and must be > 0
    prior=np.expand_dims(prior, axis=0) # for broadcasting
    prior *= args.prior_scale;
    prior = np.roll(prior, -1)
    #print(prior)
    #exit(1)

    test_generator = generator.DataGenerator(args.data, args.key_file,
                                             1, args.feat_dim, args.n_labels, False)

    path=os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
    with h5py.File(path, 'w') as f:
        for bt in range(test_generator.__len__()):
            data, keys = test_generator.__getitem__(bt, return_keys=True)
            # data = [input_sequences, label_sequences, inputs_lengths, labels_length], keys
            # return softmax outputs
            predict = model.predict_on_batch(x=data[0])
            predict = np.log(predict)
            predict -= prior # P(x|y) = P(y|x)/P(y)

            for i, key in enumerate(keys):
                # time x feats
                pr = predict[i].reshape((predict.shape[1], predict.shape[2]))
                # must be partiall
                pr = pr[:data[2][i], :]
                f.create_group(key)
                f.create_dataset(key+'/likelihood', data=pr)

                '''
                #if args.align is True:
                if False:
                    lb = data[1][i].reshape((data[1][i].shape[0],))
                    lb = lb[0:data[3][i]]
                    print(key)
                    align = dynamic_programming.dynamic_programming(pr, lb,
                                                                    data[2][i], data[3][i],
                                                                    blank=args.n_labels,
                                                                    skip_state=False)
                    align = align.reshape((1,align.shape[0]))
                    f.create_dataset(key+'/align', data=align)
                '''
                
if __name__ == "__main__":
    main()
