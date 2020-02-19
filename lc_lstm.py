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
import fixed_generator
import vgg2l
import vgg1l
import utils

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
        x=GRU(units, kernel_initializer='glorot_uniform',
                                       return_sequences=True,
                                       stateful=True,
                                       unroll=True,
                                       name='gru_fw_'+str(n))(outputs)
        # backward, not keep current states
        # do not preserve state values for backward pass
        y=GRU(units, kernel_initializer='glorot_uniform',
                                       return_sequences=True,
                                       stateful=False,
                                       unroll=True,
                                       go_backwards=True,
                                       name='lstm_bw_'+str(n))(outputs)
        outputs = Concatenate([x, y], axis=-1, name='concate_'+str(n))
        outputs=layer_normalization.LayerNormalization()(outputs)

    outputs = TimeDistributed(Dense(n_labels+1))(outputs)
    outputs = Activation('softmax')(outputs)
    outputs = Lambda(lambda x: tf.muitiply(x[0], x[1]))([outputs, masks])

    model = Model([inputs, masks], outputs)
    if optim == 'adam':
        model.compile(keras.optimizers.Adam(lr=init_lr, clipnorm=50.), loss='categorical_cross_entropy',
                    metrics='categorical_accuracy')
    else:
        model.compile(kernel.optimizers.Adadelta(lr=init_lr, clipnorm=50.), loss='categorical_cross_entropy',
                    matrics='categorical_accuracy')

    return model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='training data')
    parser.add_argument('--valid', type=str, required=True, help='validation data')
    parser.add_argument('--eval', type=str, help='evaluation data')
    parser.add_argument('--feat-dim', default=40, type=int, help='feats dim')
    parser.add_argument('--n-labels', default=1024, type=int, required=True,
                        help='number of output labels')
    parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--epochs', default=10, type=int, help='training epochs')
    parser.add_argument('--snapshot', type=str, default='./',
                        help='snapshot directory')
    parser.add_argument('--snapshot-prefix', type=str, default='snapshot',
                        help='snapshot file prefix')
    parser.add_argument('--eval-output-prefix', type=str, default='eval_out')
    parser.add_argument('--learn-rate', type=float, default=1.0e-3,
                        help='initial learn rate')
    parser.add_argument('--log-dir', type=str, default='./',
                        help='tensorboard log directory')
    parser.add_argument('--units', type=int ,default=16, help='number of LSTM cells')
    parser.add_argument('--lstm-depth', type=int ,default=2,
                        help='number of LSTM layers')
    parser.add_argument('--factor', type=float, default=0.5,help='lerarning rate decaying factor')
    parser.add_argument('--min-lr', type=float, default=1.0e-6, help='minimum learning rate')
    parser.add_argument('--process-frames', type=int, default=10, help='process frames')
    parser.add_argument('--extra-frames', type=int, default=10, help='extra frames')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate')
    parser.add_argument('--filters', type=int, default=16, help='number of filters')
    parser.add_argument('--optim', type=str, default='adadelta', help='optimizer')
    parser.add_argument('--lstm', action='store_true')
    parser.add_argument('--vgg', action='store_true')
    args = parser.parse_args()

    inputs = Input(batch_shape=(args.batch_size, None, args.feat_dim))
    masks = Input(batch_shape=(args.batch_size, None, args.feat_dim))
    model = buil_model(inputs, masks, args.units, args.lstm_depth, args.n_labels, args.feat_dim,
                       args.learn_rate, args.dropout, args.filters, args.optim, args.lstm, args.vgg):

    model = build_model(inputs, masks, args.units, args.lstm_depth, args.n_labels, args.feat_dim, args.learn_rate)

    training_generator = FixedDataGenerator(
        args.data, args.batch_size, args.feat_dim, args.n_labels,
        args.process_frames, args.extra_frames)
    valid_generator = FixedDataGenerator(
        args.valid, args.batch_size, args.feat_dim, args.n_labels,
        args.process_frames, args.extra_frames)

    tensorboard = TensorBoard(log_dir=args.log_dir)

    prev_val_acc = -1.0e10
    patience = 0
    min_val_acc = -1.0e10

    for ep in range(args.epochs):
        curr_loss = 0.0
        curr_samples=0
        curr_acc=[]

        for bt in range(training_generator.__len__()):
            # transposed---(blocks, batch, time, feats)
            x, mask, y = training_generator.__getitem__(bt)
            model.reset_states()
            for b in range(x.shape[0]):
                x_in = np.squeeze(x[b,:,:,:])
                mask_in = np.squeeze(mask[b,:,:,:])
                y_in = np.squeeze(y[b,:,:,:])
                states = get_states(model)
                loss,acc = model.train_on_batch(x=[x_in,mask_in], y=y_in)
                # for micro-mean
                samples = np.sum(mask_in)
                curr_loss += loss * samples
                curr_acc.append(acc)

                set_states(model, states)
                x_part = x_in[:, 0:args.process_frames,:]
                mask_part = mask_in[:, 0:args.process_frames,:]
                model.predict_on_batch(x=[x_part, mask_part])

            # progress report
            progress_loss = curr_loss/curr_samples
            progress_acc = np.mean(curr_acc)
            print('\rprogress: (%d/%d) loss=%.4f acc=%.4f' % (bt+1,
                training_generator.__len__(), progress_loss[0], progress_acc[0]),
                end='')
        print('\n',end='')
        curr_loss /= curr_samples
        curr_acc = np.mean(curr_acc)

        curr_val_loss = 0.0
        curr_val_acc = 0.0
        curr_val_samples = 0

        for bt in range(valid_generator.__len__()):
            x,mask,y = valid_generator.__getitem__(bt)
            model.reset_states()
            for b in range(x.shape[0]):
                x_in = np.squeeze(x[b,:,:,:])
                mask_in = np.squeeze(mask[b,:,:,:])
                y_in = np.squeeze(y[b,:,:,:])
                states = get_states(model)
                loss, acc = model.test_on_batch(x=[x_in,mask_in],y_in)

                # for micro-mean
                samples = np.sum(mask_in)
                curr_val_loss += loss * samples
                curr_val_acc.append(acc)

                set_states(model, states)
                x_part = x_in[:, 0:args.process_frames,:]
                mask_part = mask_in[:, 0:args.process_frames,:]
                model.predict_on_batch(x=[x_part, mask_part])

        print('Epoch %d (train) loss=%.4f acc=%.4f' % (ep+1, curr_loss, curr_acc))

        curr_val_loss /= curr_val_samples
        curr_val_acc = np.mean(curr_val_acc)

        if prev_val_acc > curr_val_acc:
            patience += 1
            if patience >= max_patience:
                prev_lr = K.get_value(model.optimizer.lr)
                curr_lr = prev_lr * args.factor
                if curr_lr < args.min_lr:
                    curr_lr = args.min_lr
                else:
                    print("lerning rate chaged %.4f to %.4f" % (prev_lr, curr_lr))
                    K.set_value(model.optimizer.lr,curr_lr)
                patience=0
        else:
            patience=0

        print('Epoch %d (valid) loss=%.4f acc=%.4f' % (ep+1, curr_val_loss, curr_val_acc))

        # save best model in .h5
        if min_val_acc < curr_val_acc:
            min_val_acc = curr_val_acc
            path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
            model.save_weights(path)

        prev_val_acc = curr_val_acc

if __name__ == "__main__":
    main()
