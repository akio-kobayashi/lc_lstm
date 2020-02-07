#!/bin/sh

device=$1
export CUDA_VISIBLE_DEVICES=$device
cd /home/akio/lc_lstm

direction=$2

# librispeech
train=./train.h5
valid=./dev.h5
key_file=./train.sorted
valid_key_file=./dev.sorted
n_labels=49

# features
feat_dim=40

#training
batch_size=32
epochs=100
factor=0.5
optim=adadelta
#optim=adam
dropout=0.0

for lstm_depth in 5;
do
  for units in 160;
  do
      for learn_rate in 4.0e-4;
      do
          snapdir=./model_d${lstm_depth}_d${units}_l${learn_rate}_B${batch_size}_D${dropout}_f${factor}_P3_LNtrue_BNtrue_vgg_bipolar_3_${optim}_${direction}
	  logdir=./logs_d${lstm_depth}_d${units}_l${learn_rate}_B${batch_size}_D${dropout}_f${factor}_P3_LNtrue_BNtrue_vgg_bipolar_3_${optim}_${direction}
          mkdir -p $snapdir
          mkdir -p $logdir
	      
          python ctc_lstm.py --data $train --valid $valid --key-file $key_file --valid-key-file $valid_key_file \
		 --direction ${direction} \
		 --feat-dim $feat_dim --n-labels $n_labels --batch-size $batch_size --epochs $epochs \
		 --snapshot $snapdir  --learn-rate $learn_rate --log-dir $logdir --max-patient 3\
		 --units $units --lstm-depth $lstm_depth --factor $factor \
		 --optim ${optim} --bipolar
      done
  done
done
