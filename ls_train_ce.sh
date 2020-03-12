#!/bin/sh

direction=uni
device=0
#direction=$2
#export CUDA_VISIBLE_DEVICES=$device
#cd /home/akiokobayashi0809/lc_lstm

# librispeech

#host=`hostname`
host="asr04"

n_labels=32
if [ $host == "brandy" ];then
    export CUDA_VISIBLE_DEVICES=$device
    path=ls/model_d4_d160_f32_l1.0_LNtrue_BNtrue_B16_D0.0_f0.5_vgg_lstm_adadelta_ep50_${direction}
    #cd /home/akiokobayashi0809/lc_lstm
    train=${path}/ce_train_clean_100.h5
    valid=${path}/ce_dev_clean.h5
    test=${path}/ce_test_clean.h5
    keys=${path}/ce_train_clean_100.sorted.checked
    valid_keys=${path}/ce_dev_clean.sorted.checked
#elif [ ! -e /mnt/ssd1/ ];then
#    export CUDA_VISIBLE_DEVICES=$device
    #path=ls/model_d4_d160_f32_l1.0_LNtrue_BNtrue_B16_D0.0_f0.5_vgg_lstm_adadelta_ep50_${direction}
    #cd /home/akio/lc_lstm
#    train=${path}/ce_train_clean_100.h5
#    valid=${path}/ce_dev_clean.h5
#    test=${path}/ce_test_clean.h5
#    keys=${path}/ce_train_clean_100.sorted.checked
#    valid_keys=${path}/ce_dev_clean.sorted.checked
else
    #path=ls/model_d4_d160_f32_l1.0_LNtrue_BNtrue_B16_D0.0_f0.5_vgg_lstm_adadelta_ep50_${direction}
    #root=/mnt/ssd1/eesen_20191228/eesen/asr_egs/librispeech/
    path=ls/model_d4_d256_f32_l1.0_ml1.0e-2_LNtrue_BNtrue_B16_D0.0_f0.9_vgg_lstm_adadelta_ep100_bi/
    train=${path}/ce_train.h5
    valid=${path}/ce_dev.h5
    keys=${path}/ce_train.sorted
    valid_keys=${path}/ce_dev.sorted
fi

# features
feat_dim=40
#training
batch_size=16
epochs=100
factor=0.9
dropout=0.0
optim=adadelta
filters=32

for lstm_depth in 4;
do
  for units in 256;
  do
      for learn_rate in 1.0
      do
	  #path=ls/model_d4_d256_f32_l1.0_ml1.0e-2_LNtrue_BNtrue_B16_D0.0_f0.9_vgg_lstm_adadelta_ep100_bi/
          snapdir=./ls/ce_model_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_LNtrue_BNtrue_B${batch_size}_D${dropout}_f${factor}_vgg_lstm_${optim}_${direction}
	  logdir=./ls/ce_logs_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_LNtrue_BNtrue_B${batch_size}_D${dropout}_f${factor}_vgg_lstm_${optim}_${direction}
          mkdir -p $snapdir
          mkdir -p $logdir
          python ce_lstm.py --data $train --valid $valid --direction $direction \
		 --key-file $keys --valid-key-file $valid_keys \
		 --feat-dim $feat_dim --n-labels $n_labels --batch-size $batch_size --epochs $epochs \
		 --snapshot $snapdir  --learn-rate $learn_rate --log-dir $logdir --filters ${filters} \
		 --units $units --lstm-depth $lstm_depth --factor $factor  --optim $optim --lstm 
      done
  done
done
