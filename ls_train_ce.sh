#!/bin/sh

device=$1
direction=$2
export CUDA_VISIBLE_DEVICES=$device
cd /home/akiokobayashi0809/lc_lstm

# librispeech
train=./train_ce_clean_100.h5
valid=./dev_ce_clean.h5
test=./test_ce_clean.h5
keys=./train_clean_100.sorted
valid_keys=./dev_clean.sorted
n_labels=32

# features
feat_dim=40
units=160

#training
#logs_d5_d160_l4.0e-4_LNtrue_B32_D0.0_f0.5_P3_vgg_adadelta_bi/
batch_size=32
epochs=100
factor=0.5
dropout=0.0

optim=adadelta
for lstm_depth in 5;
do
  for units in 160;
  do
      for learn_rate in 4.0e-4
      do
          snapdir=./model_ce_d${lstm_depth}_d${units}_l${learn_rate}_LNtrue_BNtrue_B${batch_size}_D${dropout}_f${factor}_vgg_${optim}_${direction}
	  logdir=./logs_ce_d${lstm_depth}_d${units}_l${learn_rate}_LNtrue_BNtrue_B${batch_size}_D${dropout}_f${factor}_vgg_${optim}_${direction}

          mkdir -p $snapdir
          mkdir -p $logdir
          python ce_lstm.py --data $train --valid $valid --direction uni --key-file $keys --valid-key-file $valid_keys \
		 --feat-dim $feat_dim --n-labels $n_labels --batch-size $batch_size --epochs $epochs \
		 --snapshot $snapdir  --learn-rate $learn_rate --log-dir $logdir \
		 --units $units --lstm-depth $lstm_depth --factor $factor  --optim $optim
      done
  done
done
