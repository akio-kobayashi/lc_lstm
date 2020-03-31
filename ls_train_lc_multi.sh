#!/bin/sh

device=0
direction=bi
#export CUDA_VISIBLE_DEVICES=$device
#cd /home/akiokobayashi0809/lc_lstm

# librispeech

host=`hostname`
n_labels=32
if [ $host == "brandy" ];then
    export CUDA_VISIBLE_DEVICES=$device
    cd /home/akiokobayashi0809/lc_lstm
    train=./ls/ce_train.h5
    valid=./ls/ce_dev.h5
    test=./ls/ce_test.h5
    keys=./ls/ce_train.sorted
    valid_keys=./ls/ce_dev.sorted
else
    path=ls/model_d4_d256_f32_l1.0_ml1.0e-2_LNtrue_BNtrue_B16_D0.0_f0.9_vgg_lstm_adadelta_ep100_bi/
    train=${path}/ce_train.h5
    valid=${path}/ce_dev.h5
    keys=${path}/ce_train.sorted
    valid_keys=${path}/ce_dev.sorted
fi

# features
feat_dim=40
batch_size=256
epochs=25
factor=0.9
dropout=0.0
optim=adadelta
filters=32
num_extra_frames1=1

for lstm_depth in 4;
do
  for units in 256;
  do
      for learn_rate in 1.0;
      do
        for proc_frames in 5 10 20 30;
        do
          for extra_frames1 in 5 10 20 30;
          do
	           if [ $proc_frames -eq $extra_frames1 ];then
		             for extra_frames2 in 5 10 20 30;
		             do
		      if [ $extra_frames1 -eq $extra_frames2 ];then
			         snapdir=./ls/lc_model_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_B${batch_size}_D${dropout}_f${factor}_LNtrue_vgg_lstm
			         snapdir=${snapdir}_p${proc_frames}_ef${extra_frames1}_es${extra_frames2}_n${num_extra_frames1}_${optim}_${direction}
			         logdir=./ls/lc_logs_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_B${batch_size}_D${dropout}_f${factor}_LNtrue_vgg_lstm
			         logdir=${logdir}_p${proc_frames}_ef${extra_frames1}_es${extra_frames2}_n${num_extra_frames1}_${optim}_${direction}
			         mkdir -p $snapdir
			         mkdir -p $logdir

			  python multi_lc_lstm.py --data $train --valid $valid \
				 --key-file $keys --valid-key-file $valid_keys \
				 --feat-dim $feat_dim \
				 --n-labels $n_labels --batch-size ${batch_size} \
				 --epochs $epochs --filters ${filters} \
				 --snapshot $snapdir  --learn-rate $learn_rate \
				 --log-dir $logdir --max-patience 3 \
				 --units $units --lstm-depth $lstm_depth \
				 --factor $factor --optim ${optim} --lstm \
				 --process-frames $proc_frames --extra-frames1 $extra_frames1 \
				 --extra-frames2 $extra_frames2 --num-extra-frames1 $num_extra_frames1
		      fi
		  done
	      fi
          done
        done
      done
  done
done
