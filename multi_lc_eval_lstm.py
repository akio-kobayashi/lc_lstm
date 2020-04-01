import argparse
import os
import sys
import subprocess
import time
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Softmax,LSTM,Activation
from keras.layers import TimeDistributed, Bidirectional, Dropout, Lambda, Masking
import keras.utils
import keras.backend as K
import numpy as np
import random
import tensorflow as tf
import multi_fixed_generator
import vgg2l
import vgg1l
import multi_utils
import layer_normalization
import network
import h5py

os.environ['PYTHONHASHSEED']='0'
np.random.seed(1024)
random.seed(1024)

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(allow_growth = True),
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)
sess = tf.Session(config=config)
keras.backend.set_session(sess)

max_label_len=1024

def get_states(model):
    return [K.get_value(s) for s,_ in model.state_updates]

def set_states(model, states):
    for (d,_), s in zip(model.state_updates, states):
        K.set_value(d, s)

def build_model(inputs, mask, units, depth, n_labels, feat_dim,
                dropout, init_filters, lstm=False, vgg=False):

    outputs = Masking(mask_value=0.0)(inputs)

    if vgg is False:
        outputs = vgg2l.VGG2L(outputs, init_filters, feat_dim)
    else:
        outputs = vgg1l.VGG(outputs, init_filters, feat_dim)

    outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, mask])
    outputs = Masking(mask_value=0.0)(outputs)

    outputs = network.lc_network(outputs, units, depth, n_labels, dropout, init_filters, lstm)
    outputs = TimeDistributed(Dense(n_labels+1))(outputs)
    outputs = Activation('softmax')(outputs)

    model = Model([inputs, mask], outputs)

    return model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='training data')
    parser.add_argument('--eval', type=str, help='evaluation data')
    parser.add_argument('--feat-dim', default=40, type=int, help='feats dim')
    parser.add_argument('--n-labels', default=1024, type=int, required=True,
                        help='number of output labels')
    parser.add_argument('--batch-size', default=1, type=int, help='mini-batch size')
    parser.add_argument('--snapshot', type=str, default='./',
                        help='snapshot directory')
    parser.add_argument('--snapshot-prefix', type=str, default='snapshot',
                        help='snapshot file prefix')
    parser.add_argument('--eval-output-prefix', type=str, default='eval_out')
    parser.add_argument('--units', type=int ,default=16, help='number of LSTM cells')
    parser.add_argument('--lstm-depth', type=int ,default=2,
                        help='number of LSTM layers')
    parser.add_argument('--process-frames', type=int, default=10, help='process frames')
    parser.add_argument('--extra-frames1', type=int, default=10, help='1st extra frames')
    parser.add_argument('--extra-frames2', type=int, default=10, help='2nd extra frames')
    parser.add_argument('--num-extra-frames1', type=int, default=1, help='number of extra frames 1')
    parser.add_argument('--filters', type=int, default=16, help='number of filters')
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--vgg', action='store_true')
    parser.add_argument('--prior', type=str, default=None, help='prior weights')
    parser.add_argument('--prior-scale', type=float, default=1.0, help='prior scaler')
    parser.add_argument('--weights', type=str, required=True, help='model weights')

    args = parser.parse_args()

    eval_in = Input(batch_shape=(args.batch_size, None, args.feat_dim))
    eval_mask = Input(batch_shape=(args.batch_size, None, args.feat_dim*args.filters*2))

    outputs = Masking(mask_value=0.0)(inputs)

    if vgg is False:
        outputs = vgg2l.VGG2L(outputs, init_filters, feat_dim)
    else:
        outputs = vgg1l.VGG(outputs, init_filters, feat_dim)

    outputs = Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, mask])
    outputs = Masking(mask_value=0.0)(outputs)

    outputs = network.lc_network(outputs, units, depth, n_labels, dropout, lstm)
    outputs = TimeDistributed(Dense(n_labels+1))(outputs)
    outputs = Activation('softmax')(outputs)

    model = Model([inputs, mask], outputs)

    return model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='training data')
    #parser.add_argument('--eval', type=str, help='evaluation data')
    parser.add_argument('--feat-dim', default=40, type=int, help='feats dim')
    parser.add_argument('--n-labels', default=1024, type=int, required=True,
                        help='number of output labels')
    parser.add_argument('--batch-size', default=1, type=int, help='mini-batch size')
    parser.add_argument('--snapshot', type=str, default='./',
                        help='snapshot directory')
    parser.add_argument('--snapshot-prefix', type=str, default='eval_snap',
                        help='snapshot file prefix')
    #parser.add_argument('--eval-output-prefix', type=str, default='eval_out')
    parser.add_argument('--units', type=int ,default=16, help='number of LSTM cells')
    parser.add_argument('--lstm-depth', type=int ,default=2,
                        help='number of LSTM layers')
    parser.add_argument('--process-frames', type=int, default=10, help='process frames')
    parser.add_argument('--extra-frames1', type=int, default=10, help='1st extra frames')
    parser.add_argument('--extra-frames2', type=int, default=10, help='2nd extra frames')
    parser.add_argument('--num-extra-frames1', type=int, default=1, help='number of extra frames 1')
    parser.add_argument('--filters', type=int, default=16, help='number of filters')
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--vgg', action='store_true')
    parser.add_argument('--prior', type=str, default=None, help='prior weights')
    parser.add_argument('--prior-scale', type=float, default=1.0, help='prior scaler')
    parser.add_argument('--weights', type=str, required=True, help='model weights')
    parser.add_argument('--dropout', type=float, default=0.0)
    args = parser.parse_args()

    eval_in = Input(batch_shape=(args.batch_size, None, args.feat_dim))
    eval_mask = Input(batch_shape=(args.batch_size, None, args.feat_dim*args.filters*2))
    #def build_model(inputs, mask, units, depth, n_labels, feat_dim,
    #            dropout, init_filters, lstm=False, vgg=False):
    eval_model = build_model(eval_in, eval_mask, args.units, args.lstm_depth, args.n_labels, args.feat_dim, args.dropout,
                             args.filters, args.lstm)
    #path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
    eval_model.load_weights(args.weights, by_name=True)

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

    eval_generator = multi_fixed_generator.FixedDataGenerator(
        args.data, None, args.batch_size, args.feat_dim, args.n_labels,
        args.process_frames, args.extra_frames1,
        args.extra_frames2, args.num_extra_frames1, mode='eval')

    path=os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
    with h5py.File(path, 'w') as f:
        for smp in range(eval_generator.__len__()):
            #processed_list=[]
            x, mask, keys = eval_generator.__getitem__(smp)
            eval_model.reset_states()
            # (block , batch, frames, fesats)
            stack=np.array(0)
            stack = None
            for b in range(x.shape[0]):
                x_in = x[b].reshape((args.batch_size, -1, args.feat_dim))
                mask_in = np.repeat(mask[b,:,:,:], args.feat_dim*args.filters*2, axis=-1)

                states = get_states(eval_model)
                predict = eval_model.predict_on_batch(x=[x_in,mask_in])
                predict = np.log(predict)
                predict -= prior
                set_states(eval_model, states)

                # another part
                x_part = x_in[:, 0:args.process_frames,:]
                mask_part = mask_in[:, 0:args.process_frames,:]
                model.predict_on_batch(x=[x_part, mask_part])
                # original part
                #mask_in[:, args.process_frames:, :]=0.0
                #eval_model.predict_on_batch(x=[x_in, mask_in])

                if b < x.shape[0]-1:
                    processed = predict[:,0:args.process_frames, :]
                else:
                    #length = int(np.sum(mask[b,:,:,:])/args.feat_dim)
                    #processed = predict[:, 0:length, args.feat_dim]
                    processd = predict
                processed = processed.reshape([-1, args.n_labels+1])

                if stack is None:
                    stack = processed
                else:
                    stack = np.vstack((stack, processed))
            for n, key in enumerate(keys):
                print(key)
                f.create_group(key)
                # check inf
                stack[np.isinf(stack)] = -99.9999
                f.create_dataset(key+'/likelihood', data=stack)
            f.flush()

if __name__ == "__main__":
    main()
