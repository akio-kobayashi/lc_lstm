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
import time
#from tf.keras.experimental import PeepholeLSTMCell
#from tf.keras.experimental import PeeholeLSTMCell
#import functools
import CTCModel
import generator
import layer_normalization

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

def build_model(inputs, units, depth, n_labels, feat_dim, init_lr, direction,
                dropout, layer_norm):

    #outputs = Masking(mask_value=0.0)(inputs)
    outputs=inputs
    for n in range (depth):
        if direction == 'bi':
            outputs=Bidirectional(CuDNNGRU(units,
                dropout=dropout,
                unit_forget_bias=True,
                return_sequences=True))(outputs)
        else:
            outputs=CuDNNGRU(units,return_sequences=True,
                dropout=dropout,
                unit_forget_bias=True)(outputs)
        if layer_norm is True:
            outputs=LayerNormalization()(outputs)
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
    outputs = Activation('softmax', name='softmax')(outputs)

    model=CTCModel.CTCModel([inputs], [outputs], greedy=False)
    model.compile(keras.optimizers.Adam(lr=init_lr, clipnorm=50.))

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
    parser.add_argument('--key-file', type=str, required=True, help='keys')
    parser.add_argument('--valid', type=str, required=True, help='validation data')
    parser.add_argument('--valid-key-file', type=str, default=None, help='validataion keys')
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
    parser.add_argument('--direction', type=str, default='bi', help='RNN direction')
    parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
    parser.add_argument('--layer-norm', type=bool, default=False, help='layer normalization')
    args = parser.parse_args()

    inputs = Input(shape=(None, args.feat_dim))
    curr_lr = args.learn_rate
    '''
    if os.path.isfile(args.log_dir+'/learn_rate'):
        with(open(args.log_dir+'/learn_rate', r)) as f:
            curr_lr=f.readline()
    '''
    model = build_model(inputs, args.units, args.lstm_depth, args.n_labels,
        args.feat_dim, curr_lr, args.direction, args.dropout, args.layer_norm)

    training_generator = generator.DataGenerator(args.data, args.key_file,
                        args.batch_size, args.feat_dim, args.n_labels, shuffle=True)
    valid_generator = generator.DataGenerator(args.valid, args.valid_key_file,
                        args.batch_size, args.feat_dim, args.n_labels, shuffle=False)

    # callbacks
    #reduce_lr = ReduceLROnPlateau(monitor='val_ler',
    #                              factor=0.5, patience=5,
    #                              min_lr=0.000001, verbose=1)
    #cp_path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
    #model_cp = ModelCheckpoint(cp_path, monitor='val_categorical_accuracy',
    #                           save_best_only=True,
    #                           save_weights_only=True, verbose=1)
    #tensorboard = TensorBoard(log_dir=args.log_dir)
    '''
    tensorboard = keras.callbacks.TensorBoard(
            log_dir=args.log_dir+'tf_logs',
            histogram_freq=0,
            batch_size=args.batch_size,
            write_graph=True,
            write_grads=True
            )
    tensorboard.set_model(model)
    '''
    prev_val_ler = 1.0e10
    patience = 0
    max_patience=5
    min_val_ler = 1.0e10
    curr_lr=args.learn_rate
    ep=0
    prev_save_ep=0
    max_early_stop=5
    early_stop=0

    '''
    if os.path.isfile(args.log_dir+'/epochs'):
        with(open(args.log_dir+'/epochs', r)) as f:
            ep=f.readline()
    if os.path.isfile(args.log_dir+'/prev_val_ler'):
        with(open(args.log_dir+'/prev_val_ler', r)) as f:
            prev_val_ler=f.readline()
    if os.path.isfile(args.log_dir+'/prev_val_ler'):
        with(open(args.log_dir+'/min_val_ler', r)) as f:
            min_val_ler=f.readline()
    '''
    with open(args.log_dir+'/logs', 'w') as logs:
        #for ep in range(args.epochs):
        while ep < args.epochs:
            start_time=time.time()
            curr_loss = 0.0
            curr_samples=0
            #curr_labels=0
            #curr_ler=[]
            for bt in range(training_generator.__len__()):
                data = training_generator.__getitem__(bt)
                # data = [input_sequences, label_sequences, inputs_lengths, labels_length]
                # y (true labels) is set to None, because not used in tensorflow CTC training.
                # 'train_on_batch' will return CTC-loss value
                loss = model.train_on_batch(x=data,y=data[1])
                # for micro-mean
                samples = data[0].shape[0]
                curr_loss += loss * samples
                curr_samples += samples
                #loss, ler, _ = model.evaluate(data)
                #curr_ler.append(ler)

                # progress report
                progress_loss = curr_loss/curr_samples
                #progress_ler = np.mean(curr_ler)*100.0
                #msg='progress: (%d/%d) loss=%.4f ler=%.4f' % (bt+1,training_generator.__len__(),
                #progress_loss, progress_ler)
                msg='progress: (%d/%d) loss=%.4f' % (bt+1,training_generator.__len__(), progress_loss)
                #print(msg,file=sys.stderr,flush=True)
                print(msg)
                logs.write(msg+'\n')

                #tensorboard.on_epoch_end(bt, named_logs(model, loss))

            logs.flush()
            #print('\n',end='',file=sys.stderr,flush=True)
            print('')
            curr_loss /= curr_samples
            #curr_ler = np.mean(curr_ler)*100.0
            curr_val_loss = 0.0
            curr_val_ler = []
            curr_val_samples = 0
            curr_val_labels = 0

            for bt in range(valid_generator.__len__()):
                data = valid_generator.__getitem__(bt)
                # eval_on_batch will return sequence error rate (ser) and label error rate (ler)
                # the function returns ['loss', 'ler', 'ser']
                # 'ler' should not be normalized by true lengths
                loss, ler, ser = model.evaluate(data)
                # for micro-mean
                samples = data[0].shape[0]
                curr_val_loss += loss[0] * samples
                curr_val_samples += samples
                curr_val_ler.append(ler)

            #msg='Epoch %d (train) loss=%.4f ler=%.4f' % (ep+1, curr_loss, curr_ler)
            msg='Epoch %d (train) loss=%.4f' % (ep+1, curr_loss)
            logs.write(msg+'\n')
            #print(msg,file=sys.stderr, flush=True)
            print(msg)
            logs.flush()

            curr_val_loss /= curr_val_samples
            curr_val_ler = np.mean(curr_val_ler)*100.0
            '''
            if prev_val_ler < curr_val_ler:
                patience += 1
                if patience >= max_patience:
                    prev_lr = K.get_value(model.model_train.optimizer.lr)
                    curr_lr = prev_lr * args.factor
                    if curr_lr < args.min_lr:
                        curr_lr = args.min_lr
                        early_stop+=1
                    else:
                        msg="lerning rate chaged %.4f to %.4f" % (prev_lr, curr_lr)
                        logs.write(msg+'\n')
                        #print(msg, file=sys.stderr,flush=True)
                        print(msg)
                        K.set_value(model.model_train.optimizer.lr,curr_lr)
                    patience=0
            else:
                patience=0
            '''
            #print('Epoch %d (valid) ler=%.4f' % (ep+1, curr_val_ler), file=sys.stderr)
            msg='Epoch %d (valid) loss=%.4f ler=%.4f' % (ep+1, curr_val_loss, curr_val_ler)
            logs.write(msg+'\n')
            #print(msg, file=sys.stderr,flush=True)
            print(msg)
            logs.flush()

            # save best model in .h5
            if min_val_ler > curr_val_ler:
                min_val_ler = curr_val_ler
                path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
                model.save_weights(path)
                prev_save_ep = ep
            else:
                if ep - prev_save_ep > max_patience:
                    prev_lr = K.get_value(model.model_train.optimizer.lr)
                    curr_lr = prev_lr * args.factor
                    if curr_lr < args.min_lr:
                        curr_lr = args.min_lr
                        early_stop+=1
                    else:
                        msg="lerning rate chaged %.4f to %.4f" % (prev_lr, curr_lr)
                        logs.write(msg+'\n')
                        print(msg)
                        K.set_value(model.model_train.optimizer.lr,curr_lr)

            if early_stop > max_early_stop:
                break

            prev_val_ler = curr_val_ler
            training_generator.on_epoch_end()
            ep += 1

            elapsed_time = time.time() - start_time
            msg="time: %.4f at epoch %d" % (elapsed_time, ep)
            logs.write(msg+'\n')
            print(msg)

            # keep stats
            '''
            with open(args.log_dir+'/epochs', 'w') as f:
                f.write(str(ep))
            with open(args.log_dir+'/learn_rate', 'w') as f:
                f.write(str(curr_lr))
            with open(args.log_dir+'/prev_val_ler', 'w') as f:
                f.write(str(prev_val_ler))
            with open(args.log_dir+'/min_val_ler', 'w') as f:
                f.write(str(min_val_ler))
            '''

    print("Training End.")
    #tensorboard.on_train_end(None)

    # evaluation
    '''
    if args.eval is not None:

        eval_in = Input(shape=(None, args.feat_dim))
        eval_model = build_model(eval_in, args.units, args.n_labels, args.feat_dim, args.learn_rate)
        path = os.path.join(args.snapshot,args.snapshot_prefix+'.h5')
        eval_model.load_weights(path, by_name=True)

        eval_generator = DataGenerator(args.eval, None, 1,
                            args.feat_dim, args.n_labels)
        path=os.path.join(args.snapshot,args.eval_out+'.h5')

        with h5py.File(path, 'w') as f:
            for smp in range(eval_generator.__len()__):
                data, keys = eval_generator.__getitem__(smp, return_keys=True)
                predict = eval_model.predict_on_batch(x=data)
                rolled=np.roll(predict, 1, axis=2) # shift for <blk>
                f.create_dataset(keys[0], data=rolled)
    '''

if __name__ == "__main__":
    main()
