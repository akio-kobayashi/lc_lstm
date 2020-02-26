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
elif [ -e /mnt/ssd1/ ];then
    root=/mnt/ssd1/eesen_20191228/eesen/asr_egs/tedlium/v1/tensorflow/
    path=model_d4_d160_f16_l1.0_B16_D0.0_f0.5_P3_LNtrue_BNtrue_vgg_adadelta_${direction}
    train=${root}/${path}/ce_train.h5
    valid=${root}/${path}/ce_dev.h5
    key_file=${root}/${path}/ce_train.sorted.checked
    valid_key_file=${root}/${path}/ce_dev.sorted.checked
else
    path=td/model_d4_d160_f16_l1.0_B16_D0.0_f0.5_P3_LNtrue_BNtrue_vgg_adadelta_${direction}
    train=${path}/ce_train.h5
    valid=${path}/ce_dev.h5
    key_file=${path}/ce_train.sorted.checked
    valid_key_file=${path}/ce_dev.sorted.checked
fi

n_labels=49

# features
feat_dim=40

#training
batch_size=128
epochs=50
factor=0.5
optim=adadelta
dropout=0.0
filters=16

for lstm_depth in 4;
do
  for units in 160;
  do
      for learn_rate in 1.0;
      do
        for proc_frames in 20 30 50;
        do
          for extra_frames in 10 20 30 50;
          do
            snapdir=./td/lc_model_d${lstm_depth}_d${units}_l${learn_rate}_B${batch_size}_D${dropout}_f${factor}_LNtrue_vgg_lstm_p${proc_frames}_e${extra_frames}_${optim}_${direction}
            logdir=./td/lc_logs_d${lstm_depth}_d${units}_l${learn_rate}_B${batch_size}_D${dropout}_f${factor}_LNtrue_vgg_lstm_p${proc_frames}_e${extra_frames}_${optim}_${direction}
            mkdir -p $snapdir
            mkdir -p $logdir

            python lc_lstm.py --data $train --valid $valid \
		          --key-file $key_file --valid-key-file $valid_key_file \
		          --feat-dim $feat_dim \
		          --n-labels $n_labels --batch-size ${batch_size} \
		          --epochs $epochs --filters ${filters} \
		          --snapshot $snapdir  --learn-rate $learn_rate \
		          --log-dir $logdir --max-patient 3 \
		          --units $units --lstm-depth $lstm_depth \
		          --factor $factor --optim ${optim} --lstm \
			  --process-frames $proc_frames --extra-frames $extra_frames
          done
        done
      done
  done
done
