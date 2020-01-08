import argparse
import os
import sys
import subprocess
import time
from keras.models import Model
from keras.layers import Dense,Input,BatchNormalization,Softmax,CuDNNLstm
from keras.layers import TimeDistributed, Bidirectional, Dropout, Lambda
import keras.utils
import keras.backend
import numpy as np
import random
import tensorflow as tf
import functools

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

def build_model(units, output_dim, feat_dim):

    input = Input(shape=(None, feat_dim),name="input")
    label_length = Input(shape=(1,), name="label_length")
    pred_length = Input(shape=(1,), name="pred_length")
    y_true = Input(shape=(max_label_len,), name="y_true")

    output = input
    for n in range (depth):
        output=Bidirectional(CuDNNLSTM(units, kernel_initializer='glorot_uniform',
                                       return_sequences=True,
                                       name='lstm_'+str(n)))(output)

    output = TimeDistributed(Dense(output_dim, name="timedist_dense"))(output)

    y_pred = Softmax()(output)

    model = Model(inputs=[input, pred_length, label_length, y_true], 
                  outputs=y_pred)

    return model, label_length, pred_length, y_true

def ctc_loss(y_true, y_pred, input_length, label_length, y_true):

    return ctc_batch_cost(y_true, y_pred, input_length, label_length)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='data')
    parser.add_argument('--valid', type=str, required=True, help='validation data')
    parser.add_argument('--feat-dim', default=40, type=int, help='feats dim')
    parser.add_argument('--output-dim', default=1024, type=int,
                        help='output class dim')
    parser.add_argument('--batch-size', default=64, type=int, help='mini-batch size')
    parser.add_argument('--epochs', default=10, type=int, help='training epochs')
    parser.add_argument('--snapshot', type=str, default='./',
                        help='snapshot directory')
    parser.add_argument('--snapshot-prefix', type=str, default='snapshot',
                        help='snapshot file prefix')
    parser.add_argument('--learn-rate', type=float, default=1.0e-3,
                        help='initial learn rate')
    parser.add_argument('--log-dir', type=str, default='./',
                        help='tensorboard log directory')
    parser.add_argument('--units', type=int ,default=16, help='number of LSTM cells')
    parser.add_argument('--lstm-depth', type=int ,default=2,
                        help='number of LSTM layers')

    args = parser.parse_args()
    
    model, label_length, pred_length, y_true = build_model(args.units,
                                                           args.output_dim,
                                                           args.feat_dim)
    ctc_loss_fn = functools.partial(ctc_loss, input_length=pred_length,
                                    label_length=label_length, y_true=y_true)

    model.compile(optimizer=keras.optimizers.Adam(lr=1.0e-4), loss=ctc_loss_fn)

    training_generator = DataGenerator(args.data,
                                          dim=(args.seq_dim, args.feat_dim),
                                          batch_size=args.batch_size)
    valid_generator = DataGenerator(args.valid,
                                    dim=(args.seq_dim, args.feat_dim),
                                    batch_size=args.batch_size)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.5, patience=5,
                                  min_lr=0.000001, verbose=1)
    cp_path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
    model_cp = ModelCheckpoint(cp_path, monitor='val_categorical_accuracy',
                               save_best_only=True,
                               save_weights_only=True, verbose=1)
    # 学習状況のログを保存する
    tensorboard = TensorBoard(log_dir=args.log_dir)
    
