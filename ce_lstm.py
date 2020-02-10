import argparse
import os
import sys
import subprocess
import time
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Softmax,LSTM,Activation, CuDNNGRU, GRU, Reshape
from keras.layers import TimeDistributed, Bidirectional, Dropout, Lambda, Masking, Conv2D
import keras.utils
import keras.backend as K
import numpy as np
import random
import tensorflow as tf
#import functools
#import CTCModel
import ce_generator
import layer_normalization

os.environ['PYTHONHASHSEED']='0'
np.random.seed(1024)
random.seed(1024)

config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(allow_growth = True),
    intra_op_parallelism_threads=1,
    inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(config=config)
keras.backend.set_session(sess)

max_label_len=1024

def build_model(inputs, mask, units, depth, n_labels, feat_dim, init_lr, direction,
                dropout, init_filters, optim):

    outputs = Masking(mask_value=0.0)(inputs)
    #outputs=inputs
    # add channel dim
    outputs=Lambda(lambda x: tf.expand_dims(x, -1))(outputs)

    filters=init_filters
    outputs=Conv2D(filters=filters,
        kernel_size=3, padding='same',
        strides=1,
        data_format='channels_last',
        kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)

    filters *= 2
    outputs=Conv2D(filters=filters,
        kernel_size=3, padding='same',
        strides=1,
        data_format='channels_last',
        kernel_initializer='glorot_uniform')(outputs)
    outputs=BatchNormalization(axis=-1)(outputs)
    outputs=Activation('relu')(outputs)

    outputs = Reshape(target_shape=(-1, feat_dim*filters))(outputs)

    outputs=Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, mask])

    for n in range (depth):
        if direction == 'bi':
            outputs=Bidirectional(CuDNNGRU(units,
                                           return_sequences=True))(outputs)
            #outputs=Bidirectional(GRU(units,
            #                          return_sequences=True))(outputs)
        else:
            outputs=CuDNNGRU(units,return_sequences=True)(outputs)
        outputs=layer_normalization.LayerNormalization()(outputs)

    outputs=Lambda(lambda x: tf.multiply(x[0], x[1]))([outputs, mask])
    outputs = Masking(mask_value=0.0)(outputs)
    outputs = TimeDistributed(Dense(n_labels+1, name="timedist_dense"))(outputs)
    outputs = Activation('softmax', name='softmax')(outputs)

    model=Model([inputs, mask], outputs)
    # we can get accuracy from data along with batch/temporal axes.
    if optim == 'adam':
        model.compile(keras.optimizers.Adam(lr=init_lr, clipnorm=50.),
            loss=['categorical_crossentropy'],
            metrics=['categorical_accuracy'])
    else:
        model.compile(keras.optimizers.Adadelta(),
            loss=['categorical_crossentropy'],
            metrics=['categorical_accuracy'])
    return model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='training data')
    parser.add_argument('--key-file', type=str, help='keys')
    parser.add_argument('--valid-key-file', type=str, help='valid keys')
    parser.add_argument('--valid', type=str, required=True, help='validation data')
    parser.add_argument('--feat-dim', default=40, type=int, help='feats dim')
    '''
    n_labels = 32 librispeech
    n_labels = 49 tedlium_v1
    '''
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
    parser.add_argument('--direction', type=str, default='bi', help='RNN direction')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--filters', type=int, default=16, help='number of filters for CNNs')
    parser.add_argument('--max-patient', type=int, default=5, help='max patient')
    parser.add_argument('--optim', type=str, default='adam', help='[adam|adadelta]')
   
    args = parser.parse_args()

    inputs = Input(shape=(None, args.feat_dim))
    mask = Input(shape=(None, 1))
    model = build_model(inputs, mask, args.units, args.lstm_depth, args.n_labels, args.feat_dim, args.learn_rate,
                        args.direction, args.dropout, args.filters, args.optim)

    training_generator = ce_generator.CEDataGenerator(args.data, args.key_file,
                                                      args.batch_size, args.feat_dim, args.n_labels, shuffle=True)
    valid_generator = ce_generator.CEDataGenerator(args.valid, args.valid_key_file,
                                                   args.batch_size, args.feat_dim, args.n_labels, shuffle=False)

    prev_val_acc = -1.0e10
    patience = 0
    max_patience=args.max_patient
    max_val_acc = -1.0e10
    curr_lr=args.learn_rate
    ep=0
    prev_save_ep=0
    max_early_stop=5
    early_stop=0

    with open(args.log_dir+'/logs', 'w') as logs:
        while ep < args.epochs:
            start_time = time.time()
            
            curr_loss = 0.0
            curr_samples=0
            curr_labels=0
            curr_acc=[]
            for bt in range(training_generator.__len__()):
                data = training_generator.__getitem__(bt)
                # data = [input_sequences, label_sequences, masks, inputs_lengths]
                #print("lengths: %d" % len(data[3]))
                if len(data[3]) == 0:
                    continue
                #print(data[0].shape)
                #print(data[1].shape)
                #print(data[2].shape)
                loss,acc = model.train_on_batch(x=[data[0],data[2]],y=data[1])

                samples = data[0].shape[0]
                curr_loss += loss * samples
                curr_samples += samples
                curr_acc.append(acc)

                # progress report
                progress_loss = curr_loss/curr_samples
                progress_acc = np.mean(curr_acc)
                msg='progress: (%d/%d) loss=%.4f acc=%.4f' % (bt+1,training_generator.__len__(),
                            progress_loss, progress_acc)
                print(msg)
                logs.write(msg+'\n')
            logs.flush()

            curr_val_loss = 0.0
            curr_val_samples = 0
            curr_val_acc = []

            ep_loss = curr_loss/curr_samples
            ep_acc = np.mean(curr_acc)
            msg='Epoch %d (train) loss=%.4f acc=%.4f' % (ep+1, ep_loss, ep_acc)
            print(msg)
            logs.write(msg+'\n')

            for bt in range(valid_generator.__len__()):
                data = valid_generator.__getitem__(bt)
                if len(data[3]) == 0:
                    continue
                loss, acc = model.test_on_batch(x=[data[0],data[2]], y=data[1])

                samples = data[0].shape[0]
                curr_val_loss += loss * samples
                curr_val_acc.append(acc)
                curr_val_samples += samples

            ep_val_loss = curr_val_loss/curr_val_samples
            ep_val_acc = np.mean(curr_val_acc)

            msg='Epoch %d (valid) loss=%.4f acc=%.4f' % (ep+1, ep_val_loss, ep_val_acc)
            print(msg)
            logs.write(msg+'\n')

            if max_val_acc < ep_val_acc:
                max_val_acc = ep_val_acc
                path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
                model.save_weights(path)
                msg="save the model epoch %d" % (ep+1)
                logs.write(msg+'\n')
                print(msg)
                logs.flush()
                prev_save_ep = ep
            else:
                if ep - prev_save_ep > max_patience:
                    prev_lr = K.get_value(model.model_train.optimizer.lr)
                    curr_lr = prev_lr * args.factor
                    if curr_lr < args.min_lr:
                        curr_lr = args.min_lr
                        early_stop+=1
                    else:
                        msg="learning rate chaged %.4f to %.4f at epoch %d" % (prev_lr, curr_lr, ep+1)
                        logs.write(msg+'\n')
                        print(msg)
                        K.set_value(model.model_train.optimizer.lr,curr_lr)

            if early_stop > max_early_stop:
                break

            prev_val_acc = ep_val_acc
            training_generator.on_epoch_end()
            ep += 1

            elapsed_time = time.time() - start_time
            msg="time: %.4f at epoch %d" % (elapsed_time, ep)
            logs.write(msg+'\n')
            print(msg)


if __name__ == "__main__":
    main()
