#!/bin/sh

device=$1
direction=$2
export CUDA_VISIBLE_DEVICES=$device
cd /home/akiokobayashi0809/lc_lstm

# librispeech
train=./train_clean_100.h5
valid=./dev_clean.h5
test=./test_clean.h5
keys=./train_clean_100.sorted
valid_keys=./dev_clean.sorted
n_labels=32

# features
feat_dim=40
units=160

#training
batch_size=32
epochs=50
factor=0.5
dropout=0.0
filters=32

for lstm_depth in 5;
do
  for units in 160;
  do
      for learn_rate in 1.0;
      do
	  for optim in adadelta;
	  do
              snapdir=./model_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_LNtrue_BNtrue_B${batch_size}_D${dropout}_f${factor}_vgg_${optim}_ep${epochs}_${direction}
	      logdir=./logs_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_LNtrue_BNtrue_B${batch_size}_D${dropout}_f${factor}_vgg_${optim}_ep${epochs}_${direction}
	      
              mkdir -p $snapdir
              mkdir -p $logdir
              python ctc_lstm.py --data $train --valid $valid --direction uni --key-file $keys \
		     --valid-key-file $valid_keys \
		     --feat-dim $feat_dim --n-labels $n_labels --batch-size $batch_size --epochs $epochs \
		     --snapshot $snapdir  --learn-rate $learn_rate --log-dir $logdir \
		     --units $units --lstm-depth $lstm_depth --factor $factor  --optim $optim --filters $filters
	  done
      done
  done
done

