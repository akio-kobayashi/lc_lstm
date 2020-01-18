#!/bin/sh

device=$1
export CUDA_VISIBLE_DEVICES=$device
cd ~/lc_lstm

# librispeech
train=./train_clean_100.h5
valid=./dev_clean.h5
keys=./train_clean_100.sorted
n_labels=32

# features
feat_dim=40
units=320
lstm_depth=3

#training
#direction=bi
batch_size=128
epochs=50
learn_rate=1.0e-3
factor=0.9

for lstm_depth in 3 4 5;
do
  for units in 320;
  do
    for learn_rate in 1.0e-4 4.0e-5 1.0e-5;
    do
      if [ $device == 1 ];then
        snapdir=./model_d${lstm_depth}_d${units}_l${learn_rate}_uni
        logdir=./logs_d${lstm_depth}_d${units}_l${learn_rate}_uni
        mkdir -p $snapdir
        mkdir -p $logdir
        python ctc_lstm.py --data $train --valid $valid --direction uni \
        --feat-dim $feat_dim --n-labels $n_labels --batch-size $batch_size --epochs $epochs \
        --snapshot $snapdir  --learn-rate $learn_rate --log-dir $logdir \
        --units $units --lstm-depth $lstm_depth --factor $factor
      else
        snapdir=./model_d${lstm_depth}_d${units}_l${learn_rate}_bi
        logdir=./logs_d${lstm_depth}_d${units}_l${learn_rate}_bi
        mkdir -p $snapdir
        mkdir -p $logdir
        python ctc_lstm.py --data $train --valid $valid --direction bi \
        --feat-dim $feat_dim --n-labels $n_labels --batch-size $batch_size --epochs $epochs \
        --snapshot $snapdir  --learn-rate $learn_rate --log-dir $logdir \
        --units $units --lstm-depth $lstm_depth --factor $factor
      fi
    done
  done
done
