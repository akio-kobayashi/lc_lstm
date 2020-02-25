import argparse
import os
import sys
import subprocess
import time
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Softmax,LSTM,Activation
from keras.layers import TimeDistributed, Bidirectional, Dropout, Lambda, Masking
import keras.utils
import keras.backend
import numpy as np
import random
import tensorflow as tf
import multi_fixed_generator
import vgg2l
import vgg1l
import multi_utils
import layer_normalization

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

def build_model(inputs, mask, units, depth, n_labels, feat_dim, init_lr,
                dropout, init_filters, optim, lstm=False, vgg=False):

    outputs = Masking(mask_value=0.0)(inputs)

    if vgg is False:
        outputs = vgg2l.VGG2L(outputs, init_filters, feat_dim)
    else:
        outputs = vgg1l.VGG(outputs, init_filters, feat_dim)

    for n in range (depth):
        # forward, keep current states
        # statefule
        if lstm is False:
            x=GRU(units, kernel_initializer='glorot_uniform',
                  return_sequences=True,
                  stateful=True,
                  unroll=False)(outputs)
        else:
            x=LSTM(units, kernel_initializer='glorot_uniform',
                  return_sequences=True,
                  stateful=True,
                  unroll=False)(outputs)
        # backward, not keep current states
        # do not preserve state values for backward pass
        if lstm is False:
            y=GRU(units, kernel_initializer='glorot_uniform',
                  return_sequences=True,
                  stateful=False,
                  unroll=False,
                  go_backwards=True)(outputs)
        else:
            y=LSTM(units, kernel_initializer='glorot_uniform',
                   return_sequences=True,
                   stateful=False,
                   unroll=False,
                   go_backwards=True)(outputs)
        outputs = Concatenate(axis=-1)([x,y])
        outputs=layer_normalization.LayerNormalization()(outputs)

    outputs = TimeDistributed(Dense(n_labels+1))(outputs)
    outputs = Activation('softmax')(outputs)

    model = Model([inputs, mask], outputs)
    if optim == 'adam':
        model.compile(keras.optimizers.Adam(lr=init_lr, clipnorm=50.), loss='categorical_crossentropy',
                    metrics='categorical_accuracy')
    else:
        model.compile(keras.optimizers.Adadelta(lr=init_lr, clipnorm=50.), loss='categorical_crossentropy',
                    matrics='categorical_accuracy')

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
    args = parser.parse_args()

    eval_in = Input(batch_shape=(args.batch_size, None, args.feat_dim))
    eval_mask = Input(batch_shape=(args.batch_size, None, args.feat_dim))
    eval_model = build_model(eval_in, eval_mask, args.units, args.n_labels, args.feat_dim, args.learn_rate)
    path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
    eval_model.load_weights(path, by_name=True)

    eval_generator = multi_fixed_genrator.FixedDataGenerator(
        args.eval, args.batch_size, args.feat_dim, args.n_labels,
        args.process_frames, args.extra_frames1,
        args.extra_frames2, args.num_extra_frames1, mode='eval')

    path=os.path.join(args.snapshot,args.eval_out+'.ark')
    with h5py.File(path, 'w') as f:
        for smp in range(eval_generator.__len()__):
            processed_list=[]
            x, mask, keys = eval_generator.__getitem__(smp)
            model.reset_states()
            # (block , batch, frames, fesats)
            stack=np.array()
            for b in range(x.shape[0]):
                x_in = x.reshape((args.batch_size, -1, args.feat_dim))
                mask_in = mask.reshape((args.batch_size, -1, args.feat_dim))
                states = get_states(model)
                predict = eval_model.predict_on_batch(x=[x_in,mask_in])
                set_states(eval_model, states)
                x_part = x_in[:, 0:args.process_frames,:]
                mask_part = mask_in[:, 0:args.process_frames,:]
                eval_model.predict_on_batch(x=[x_part, mask_part])

                if b < x.shape[0]-1:
                    processed = predict[:,0:args.process_frames, :]
                else:
                    length = int(np.sum(mask_in)/args.feat_dim)
                    processed = predict[:, 0:length, feat_dim]
                processed = processed.reshape([-1, args.feat_dim])
                if len(stack) == 0:
                    stack = processed
                else:
                    stack = np.vstack(stack, processed)
            for n, key in enumerate(keys):
                f.create_group(keys[0])
                f.create_dataset(key+'/likelihood', data=stack)

if __name__ == "__main__":
    main()
