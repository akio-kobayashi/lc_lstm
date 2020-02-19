#!/bin/sh

device=$1
direction=$2

export CUDA_VISIBLE_DEVICES=$device
#cd /home/akio/lc_lstm
cd /home/akiokobayashi0809/lc_lstm
#cd ~/lc_lstm

# librispeech
train=./js/train_jnas.h5
valid=./js/dev_jnas.h5
keys=./js/train_jnas.sorted
valid_keys=./js/dev_jnas.sorted
n_labels=2363

# features
feat_dim=40
units=160
#training
batch_size=16
epochs=100
factor=0.5
dropout=0.0
optim=adadelta
filters=32

for lstm_depth in 4;
do
  for units in 160;
  do
    for learn_rate in 1.0;
    do
      snapdir=./js/model_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_LNtrue_BNtrue_B${batch_size}_D${dropout}_f${factor}_vgg_${optim}_${direction}
      logdir=./js/logs_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_LNtrue_BNtrue_B${batch_size}_D${dropout}_f${factor}_vgg_lstm_${optim}_${direction}

      if [ ! -e $snapdir/snapshot.h5 ]; then
        mkdir -p $snapdir
        mkdir -p $logdir
        python ctc_lstm.py --data $train --valid $valid --direction $direction --key-file $keys --valid-key-file $valid_keys \
          --feat-dim $feat_dim --n-labels $n_labels --batch-size $batch_size --epochs $epochs \
          --snapshot $snapdir  --learn-rate $learn_rate --log-dir $logdir \
          --units $units --lstm-depth $lstm_depth --factor $factor --dropout $dropout \
          --optim $optim --filters $filters --lstm
	    fi
    done
  done
done
