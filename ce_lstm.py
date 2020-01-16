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
import ce_generator

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

def build_model(inputs, units, depth, n_labels, feat_dim, init_lr):

    outputs = Masking(mask_value=0.0)(inputs)
    for n in range (depth):
        outputs=Bidirectional(LSTM(units, kernel_initializer='glorot_uniform',
                                       return_sequences=True,
                                       unit_forget_bias=True,
                                       name='lstm_'+str(n)))(outputs)

    outputs = TimeDistributed(Dense(n_labels+1, name="timedist_dense"))(outputs)
    outputs = Activation('softmax', name='softmax')(outputs)

    model=Model(inputs, outputs)
    # we can get accuracy from data along with batch/temporal axes.
    model.compile(keras.optimizers.Adam(lr=init_lr),
        loss=['categorical_cross_entropy'],
        metrics=['categorical_accuracy'])

    return model

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='training data')
    parser.add_argument('--key-file', type=str, help='keys')
    parser.add_argument('--valid', type=str, required=True, help='validation data')
    parser.add_argument('--eval', type=str, help='evaluation data')
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

    args = parser.parse_args()

    inputs = Input(shape=(None, args.feat_dim))
    model = build_model(inputs, args.units, args.lstm_depth, args.n_labels, args.feat_dim, args.learn_rate)

    training_generator = generator.CEDataGenerator(args.data, args.key_file,
                        args.batch_size, args.feat_dim, args.n_labels)
    valid_generator = generator.CEDataGenerator(args.valid, None,
                        args.batch_size, args.feat_dim, args.n_labels)

    # callbacks
    #reduce_lr = ReduceLROnPlateau(monitor='val_ler',
    #                              factor=0.5, patience=5,
    #                              min_lr=0.000001, verbose=1)
    #cp_path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
    #model_cp = ModelCheckpoint(cp_path, monitor='val_categorical_accuracy',
    #                           save_best_only=True,
    #                           save_weights_only=True, verbose=1)
    #tensorboard = TensorBoard(log_dir=args.log_dir)

    prev_val_acc = 0.0
    patience = 0
    max_val_acc = 0.0

    for ep in range(args.epochs):
        curr_loss = 0.0
        curr_samples=0
        curr_labels=0
        curr_acc=0.0
        print('progress:')
        for bt in range(training_generator.__len__()):
            data = training_generator.__getitem__(bt)
            # data = [input_sequences, label_sequences, inputs_lengths]
            loss,acc = model.train_on_batch(x=data[0],y=data[1])
            # for micro-mean
            samples = data[0].shape[0]
            curr_loss += loss * samples
            curr_acc += np.sum(np.array(acc))*samples
            curr_samples += samples

            # progress report
            progress_loss = curr_loss/curr_samples
            progress_acc = curr_acc*100.0/curr_samples
            print('\rprogress: (%d/%d) loss=%.4f acc=%.4f' % (bt+1,
                training_generator.__len__(), progress_loss, progress_acc),
                end='')
        print('\n',end='')
        curr_loss /= curr_samples
        curr_acc = curr_acc*100.0/curr_samples
        curr_val_loss = 0.0
        curr_val_acc = 0.0
        curr_val_samples = 0

        for bt in range(valid_generator.__len__()):
            data = valid_generator.__getitem__(bt)
            loss, acc = model.test_on_batch(x=data[0], y=data[1])
            # for micro-mean
            samples = data[0].shape[0]
            curr_val_loss += loss * samples
            curr_val_acc += np.sum(np.array(acc))*samples
            curr_val_samples += samples

        print('Epoch %d (train) loss=%.4f acc=%.4f' % (ep+1, curr_loss, curr_acc))

        curr_val_loss /= curr_val_samples
        curr_val_acc = curr_val_acc*100.0/curr_val_samples
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
        if max_val_acc < curr_val_acc:
            max_val_acc = curr_val_acc
            path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
            model.save_weights(path)

        prev_val_acc = curr_val_acc

    # evaluation
    '''
    if args.eval is not None:

        eval_in = Input(shape=(None, args.feat_dim))
        eval_model = build_model(eval_in, args.units, args.n_labels, args.feat_dim, args.learn_rate)
        path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
        eval_model.load_weights(path, by_name=True)

        eval_generator = CEDataGenerator(args.eval, None, 1,
                            args.feat_dim, args.n_labels)
        path=os.path.join(args.snapshot,args.eval_out+'.h5')

        with h5py.File(path, 'w') as f:
            for smp in range(eval_generator.__len()__):
                data, keys = eval_generator.__getitem__(smp, return_keys=True)
                predict = eval_model.predict_on_batch(x=data[0])
                rolled=np.roll(predict, 1, axis=2) # shift for <blk>
                f.create_dataset(keys[0], data=rolled)
    '''

if __name__ == "__main__":
    main()