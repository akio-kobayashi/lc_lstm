#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
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
batch_size=16
epochs=50
learn_rate=1.0e-4
factor=0.9

mkdir -p ./snaps
mkdir -p ./logs
python ctc_lstm.py --data $train --key-file $keys --valid $valid \
  --feat-dim $feat_dim --n-labels $n_labels --batch-size $batch_size --epochs $epochs \
  --snapshot ./snaps  --learn-rate $learn_rate --log-dir ./logs \
  --units $units --lstm-depth $lstm_depth --factor $factor
