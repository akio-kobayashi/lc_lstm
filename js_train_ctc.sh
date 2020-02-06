#!/bin/sh

device=$1
direction=$2

export CUDA_VISIBLE_DEVICES=$device
#cd /home/akio/lc_lstm
cd /home/akiokobayashi0809/lc_lstm
#cd ~/lc_lstm

# librispeech
train=./train_jnas.h5
valid=./dev_jnas.h5
keys=./train_jnas.sorted
valid_keys=./dev_jnas.sorted
n_labels=2363

# features
feat_dim=40
units=160

#training
batch_size=32
epochs=100
factor=0.5

dropout=0.0
optim=adadelta

for lstm_depth in 4;
do
  for units in 160;
  do
    for learn_rate in 4.0e-4;
    do
        snapdir=./js_model_d${lstm_depth}_d${units}_l${learn_rate}_LNtrue_BNtrue_B${batch_size}_D${dropout}_f${factor}_P3_vgg_${optim}_${direction}
        logdir=./js_logs_d${lstm_depth}_d${units}_l${learn_rate}_LNtrue_BNtrue_B${batch_size}_D${dropout}_f${factor}_P3_vgg_${optim}_${direction}

	if [ ! -e $snapdir/snapshot.h5 ]; then
            mkdir -p $snapdir
            mkdir -p $logdir
            python ctc_lstm.py --data $train --valid $valid --direction uni --key-file $keys --valid-key-file $valid_keys \
		   --feat-dim $feat_dim --n-labels $n_labels --batch-size $batch_size --epochs $epochs \
		   --snapshot $snapdir  --learn-rate $learn_rate --log-dir $logdir \
		   --units $units --lstm-depth $lstm_depth --factor $factor --dropout $dropout \
		   --max-patient 3 --optim ${optim}
	fi
    done
  done
done
