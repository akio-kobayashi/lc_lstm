#!/bin/sh
device=$1
export CUDA_VISIBLE_DEVICES=$device
cd /home/akio/lc_lstm

direction=$2

host=`hostname`
if [ $host == "brandy" ];then
    export CUDA_VISIBLE_DEVICES=$device
    cd /home/akiokobayashi0809/lc_lstm
    train=./train.h5
    valid=./dev.h5
    key_file=./train.sorted
    valid_key_file=./dev.sorted
elif [ $host == "asr03" ];then
    export CUDA_VISIBLE_DEVICES=$device
    cd /home/akio/lc_lstm
    train=./train.h5
    valid=./dev.h5
    key_file=./train.sorted
    valid_key_file=./dev.sorted
else
    root=/mnt/ssd1/eesen_20191228/eesen/asr_egs/tedlium/v1/data/
    train=${root}/train/train.h5
    valid=${root}/dev/dev.h5
    key_file=${root}/train/train.sorted
    valid_key_file=${root}/dev/dev.sorted
fi

n_labels=49

# features
feat_dim=40
#training
batch_size=16
epochs=100
factor=0.9
optim=adadelta
#optim=adam
dropout=0.0
filters=32

for lstm_depth in 3;
do
  for units in 256;
  do
      for learn_rate in 1.0;
      do
          snapdir=./td/model_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_B${batch_size}_D${dropout}_f${factor}_LNtrue_BNtrue_vgg_lstm_${optim}_ep${epochs}_${direction}
	  logdir=./td/logs_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_B${batch_size}_D${dropout}_f${factor}_P3_LNtrue_BNtrue_vgg_lstm_${optim}_ep${epochs}_${direction}
          mkdir -p $snapdir
          mkdir -p $logdir
	      
          python ctc_lstm.py --data $train --valid $valid --key-file $key_file --valid-key-file $valid_key_file \
		 --direction ${direction} \
		 --feat-dim $feat_dim --n-labels $n_labels --batch-size $batch_size --epochs $epochs \
		 --snapshot $snapdir  --learn-rate $learn_rate --log-dir $logdir --max-patience 3\
		 --units $units --lstm-depth $lstm_depth --factor $factor \
		 --optim ${optim} --filters ${filters} 
      done
  done
done
