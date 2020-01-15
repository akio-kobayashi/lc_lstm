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
#import functools
#import CTCModel
import fixed_generator
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

def build_model(inputs, masks, units, n_labels, feat_dim, init_lr):

    outputs = Masking(mask_value=0.0)(inputs)
    for n in range (depth):
        # forward, keep current states
        x=LSTM(units, kernel_initializer='glorot_uniform',
                                       return_sequences=True,
                                       unit_forget_bias=True,
                                       stateful=True,
                                       unroll=True,
                                       name='lstm_fw_'+str(n))(outputs)
        # backward, not keep current states
        y=LSTM(units, kernel_initializer='glorot_uniform',
                                       return_sequences=True,
                                       unit_forget_bias=True,
                                       stateful=False,
                                       unroll=True,
                                       go_backwards=True,
                                       name='lstm_bw_'+str(n))(outputs)
        outputs = Concatenate([x, y], axis=-1, name='concate_'+str(n))

    outputs = TimeDistributed(Dense(n_labels+1, name="timedist_dense"))(outputs)
    outputs = Activation('softmax', name='softmax')(outputs)
    outputs = tf.muitiply(outputs, masks)

    model = Model([inputs, masks], outputs)
    model.compile(keras.optimizers.Adam(lr=init_lr), loss='categorical_cross_entropy',
                metrics='categorical_accuracy')

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
    args = parser.parse_args()

    inputs = Input(batch_shape=(args.batch_size, None, args.feat_dim))
    masks = Input(batch_shape=(args.batch_size, None, args.feat_dim))
    model = build_model(inputs, masks, args.units, args.n_labels, args.feat_dim, args.learn_rate)

    training_generator = FixedDataGenerator(
        args.data, args.batch_size, args.feat_dim, args.n_labels,
        args.process_frames, args.extra_frames)
    valid_generator = FixedDataGenerator(
        args.valid, args.batch_size, args.feat_dim, args.n_labels,
        args.process_frames, args.extra_frames)
    # callbacks
    #reduce_lr = ReduceLROnPlateau(monitor='val_ler',
    #                              factor=0.5, patience=5,
    #                              min_lr=0.000001, verbose=1)
    #cp_path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
    #model_cp = ModelCheckpoint(cp_path, monitor='val_categorical_accuracy',
    #                           save_best_only=True,
    #                           save_weights_only=True, verbose=1)
    tensorboard = TensorBoard(log_dir=args.log_dir)

    prev_val_acc = -1.0e10
    patience = 0
    min_val_acc = -1.0e10

    for ep in range(args.epochs):
        curr_loss = 0.0
        curr_samples=0
        print('progress:')
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
                curr_acc += acc * samples;
                curr_samples += samples

                set_states(model, states)
                x_part = x_in[:, 0:args.process_frames,:]
                mask_part = mask_in[:, 0:args.process_frames,:]
                model.predict_on_batch(x=[x_part, mask_part])

            # progress report
            progress_loss = curr_loss/curr_samples
            progress_acc = curr_acc/curr_samples
            print('\rprogress: (%d/%d) loss=%.4f acc=%.4f' % bt+1,
                training_generator.__len__(), progress_loss, progress_acc,
                end='')
        print('\n',end='')
        curr_loss /= curr_samples
        curr_acc = curr_acc*100.0/curr_labels

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
                curr_val_ler += acc * samples
                curr_val_samples += samples

                set_states(model, states)
                x_part = x_in[:, 0:args.process_frames,:]
                mask_part = mask_in[:, 0:args.process_frames,:]
                model.predict_on_batch(x=[x_part, mask_part])

        print('Epoch %d (train) loss=%.4f acc=%.4f' % ep+1, curr_loss, curr_acc)

        curr_val_loss /= curr_val_samples
        curr_val_ler = curr_val_ler*100.0/curr_val_samples
        if prev_val_acc > curr_val_acc:
            patience += 1
            if patience >= max_patience:
                prev_lr = K.get_value(model.optimizer.lr)
                curr_lr = prev_lr * args.factor
                if curr_lr < args.min_lr:
                    curr_lr = args.min_lr
                else:
                    print("lerning rate chaged %.4f to %.4f" % prev_lr, curr_lr)
                    K.set_value(model.optimizer.lr,curr_lr)
                patience=0
        else:
            patience=0

        print('Epoch %d (valid) loss=%.4f acc=%.4f' % ep+1, curr_val_loss, curr_val_acc)

        # save best model in .h5
        if min_val_acc < curr_val_acc:
            min_val_acc = curr_val_acc
            path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
            model.save_weights(path)

        prev_val_acc = curr_val_acc

    # evaluation
    '''
    if args.eval is not None:

        eval_in = Input(batch_shape=(1, None, args.feat_dim))
        eval_mask = Input(batch_shape=(1, None, args.feat_dim))
        eval_model = build_model(eval_in, eval_mask, args.units, args.n_labels, args.feat_dim, args.learn_rate)
        path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
        eval_model.load_weights(path, by_name=True)

        eval_generator = DataGenerator(
            args.eval, 1, args.feat_dim, args.n_labels,
            args.process_frames, args.extra_frames)
        path=os.path.join(args.snapshot,args.eval_out+'.ark')

        with h5py.File(path, 'w') as f:
            for smp in range(eval_generator.__len()__):
                x, mask, y, keys = eval_generator.__getitem__(smp)
                model.reset_states()
                for b in range(x.shape[0]):
                    x_in = np.squeeze(x[b,:,:,:])
                    mask_in = np.squeeze(mask[b,:,:,:])
                    y_in = np.squeeze(y[b,:,:,:])
                    states = get_states(model)
                    predict = eval_model.predict_on_batch(x=[x_in,mask_in])
                    set_states(eval_model, states)
                    #
                    x_part = x_in[:, 0:args.process_frames, :]
                    mask_part = mask_in[:, 0:args.process_frames, :]

                f.create_dataset(keys[0], predict)
    '''

if __name__ == "__main__":
    main()
