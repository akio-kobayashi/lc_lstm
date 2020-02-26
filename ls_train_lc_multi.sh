#!/bin/sh

device=$1
direction=$2
#export CUDA_VISIBLE_DEVICES=$device
#cd /home/akiokobayashi0809/lc_lstm

# librispeech

host=`hostname`
n_labels=32
if [ $host == "brandy" ];then
    export CUDA_VISIBLE_DEVICES=$device
    path=ls/model_d4_d160_f32_l1.0_LNtrue_BNtrue_B16_D0.0_f0.5_vgg_lstm_adadelta_ep50_${direction}
    cd /home/akiokobayashi0809/lc_lstm
    train=${path}/ce_train_clean_100.h5
    valid=${path}/ce_dev_clean.h5
    test=${path}/ce_test_clean.h5
    keys=${path}/ce_train_clean_100.sorted.checked
    valid_keys=${path}/ce_dev_clean.sorted.checked
elif [ ! -e /mnt/ssd1/ ];then
    export CUDA_VISIBLE_DEVICES=$device
    path=ls/model_d4_d160_f32_l1.0_LNtrue_BNtrue_B16_D0.0_f0.5_vgg_lstm_adadelta_ep50_${direction}
    cd /home/akio/lc_lstm
    train=${path}/ce_train_clean_100.h5
    valid=${path}/ce_dev_clean.h5
    test=${path}/ce_test_clean.h5
    keys=${path}/ce_train_clean_100.sorted.checked
    valid_keys=${path}/ce_dev_clean.sorted.checked
else
    path=ls/model_d4_d160_f32_l1.0_LNtrue_BNtrue_B16_D0.0_f0.5_vgg_lstm_adadelta_ep50_${direction}
    root=/mnt/ssd1/eesen_20191228/eesen/asr_egs/librispeech/
    train=${root}/${path}/ce_train_clean_100.h5
    valid=${root}/${path}/ce_dev_clean.h5
    keys=${root}/${path}/ce_train_clean_100.sorted.checked
    valid_keys=${root}/${path}/ce_dev_clean.sorted.checked
fi

# features
feat_dim=40
units=160
batch_size=256
epochs=50
factor=0.9
dropout=0.0
optim=adadelta
filters=32
num_extra_frames1=1

for lstm_depth in 4;
do
  for units in 160;
  do
      for learn_rate in 1.0;
      do
        for proc_frames in 50;
        do
          for extra_frames1 in 20:
          do
            for extra_frames2 in 10;
            do
              if [ $extra_frames1 -gt $extra_frames2 ]; then
                snapdir=./ls/lc_model_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_B${batch_size}_D${dropout}_f${factor}_LNtrue_vgg_lstm_p${proc_frames}_e1${extra_frames1}_e2${extra_frames2}_${optim}_${direction}
                logdir=./ls/lc_logs_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_B${batch_size}_D${dropout}_f${factor}_LNtrue_vgg_lstm_p${proc_frames}_e1${extra_frames1}_e2${extra_frames2}_${optim}_${direction}
                mkdir -p $snapdir
                mkdir -p $logdir

                python lc_lstm_multi.py --data $train --valid $valid \
		                --key-file $keys --valid-key-file $valid_keys \
		                --feat-dim $feat_dim \
		                --n-labels $n_labels --batch-size ${batch_size} \
		                --epochs $epochs --filters ${filters} \
		                --snapshot $snapdir  --learn-rate $learn_rate \
		                --log-dir $logdir --max-patient 3 \
		                --units $units --lstm-depth $lstm_depth \
		                --factor $factor --optim ${optim} --lstm \
			              --process-frames $proc_frames --extra-frames1 $extra_frames1 \
                    --extra_frames2 $extra_frames2 --num-extra-frames1 $num_extra_frames1
              fi
            done
          done
        done
      done
  done
done
