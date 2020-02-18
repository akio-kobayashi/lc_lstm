#!/bin/sh

direction=bi
host=`hostname`
if [ $host == "brandy" ];then
    device=1
    export CUDA_VISIBLE_DEVICES=$device
    cd /home/akio/lc_lstm
    # librispeech
    train=./train.h5
    valid=./dev.h5
    key_file=./train.sorted
    valid_key_file=./dev.sorted
elif [ -e /mnt/ssd1/eesen_20191228/eesen/asr_egs/tedlium/v1/tensorflow/ ];then
    root=/mnt/ssd1/eesen_20191228/eesen/asr_egs/tedlium/v1/tensorflow/
    path=model_d4_d160_f16_l1.0_B16_D0.0_f0.5_P3_LNtrue_BNtrue_vgg_adadelta_bi
    train=${root}/${path}/ce_train.h5
    valid=${root}/${path}/ce_dev.h5
    key_file=${root}/${path}/ce_train.sorted.checked
    valid_key_file=${root}/${path}/ce_dev.sorted.checked
fi

n_labels=49

# features
feat_dim=40

#training
batch_size=16
epochs=100
factor=0.5
optim=adadelta
dropout=0.0

for lstm_depth in 5;
do
  for units in 160;
  do
      for learn_rate in 2.0e-4;
      do
         snapdir=./model_ce_d${lstm_depth}_d${units}_l${learn_rate}_B${batch_size}_D${dropout}_f${factor}_P3_LNtrue_vgg_${optim}_${direction}
         logdir=./logs_ce_d${lstm_depth}_d${units}_l${learn_rate}_B${batch_size}_D${dropout}_f${factor}_P3_LNtrue_vgg_${optim}_${direction}
         mkdir -p $snapdir
         mkdir -p $logdir

         python ce_lstm.py --data $train --valid $valid --key-file $key_file --valid-key-file $valid_key_file \
		--direction bi --feat-dim $feat_dim --n-labels $n_labels --batch-size 32 \
		--epochs $epochs \
		--snapshot $snapdir  --learn-rate $learn_rate --log-dir $logdir --max-patient 3 \
		--units $units --lstm-depth $lstm_depth --factor $factor --optim ${optim}
      done
  done
done
