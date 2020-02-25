#!/bin/sh

device=$1
direction=$2

host=`hostname`
if [ $host == "brandy" ];then
    export CUDA_VISIBLE_DEVICES=$device
    cd /home/akiokobayashi0809/lc_lstm
    train=./train_clean_100.h5
    valid=./dev_clean.h5
    test=./test_clean.h5
    keys=./train_clean_100.sorted
    valid_keys=./dev_clean.sorted
elif [ ! -e /mnt/ssd1/ ];then
    export CUDA_VISIBLE_DEVICES=$device
    cd /home/akio/lc_lstm
    train=./train_clean_100.h5
    valid=./dev_clean.h5
    test=./test_clean.h5
    keys=./train_clean_100.sorted
    valid_keys=./dev_clean.sorted
else
    root=/mnt/ssd1/eesen_20191228/eesen/asr_egs/librispeech/
    train=${root}/exp/nml_seq_fw_seq_tw/train_clean_100/train_clean_100.h5
    valid=${root}/exp/nml_seq_fw_seq_tw/dev_clean/dev_clean.h5
    keys=${root}/exp/nml_seq_fw_seq_tw/train_clean_100/train_clean_100.sorted
    valid_keys=${root}/exp/nml_seq_fw_seq_tw/dev_clean/dev_clean.sorted
fi

export CUDA_VISIBLE_DEVICES=$device
n_labels=32
prior=./ls_label_counts.h5

# features
feat_dim=40
units=160
lstm_depth=4
batch_size=16
learn_rate=1.0
factor=0.5
dropout=0.0
optim=adadelta
#filters=16
filters=32
epochs=50

# model_d4_d160_f32_l1.0_LNtrue_BNtrue_B16_D0.0_f0.5_vgg1l_adadelta_ep50_bi
snapdir=./ls/model_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_LNtrue_BNtrue_B${batch_size}_D${dropout}_f${factor}_vgg_${optim}_ep${epochs}_${direction}
weights=${snapdir}/snapshot.h5

if [ -e $weights ]; then
    python eval_ctc_lstm.py --data $train --key-file $keys --direction ${direction} \
	   --feat-dim $feat_dim --n-labels $n_labels \
	   --snapshot $snapdir  --snapshot-prefix train --weight $weights \
	   --units $units --lstm-depth $lstm_depth  --filters ${filters} #--lstm

    python eval_ctc_lstm.py --data $valid --key-file $valid_keys --direction ${direction} \
	   --feat-dim $feat_dim --n-labels $n_labels \
	   --snapshot $snapdir  --snapshot-prefix dev --weight $weights \
	   --units $units --lstm-depth $lstm_depth  --filters ${filters} #--lstm

    python eval_ctc_lstm.py --data $test --direction ${direction} \
	   --feat-dim $feat_dim --n-labels $n_labels \
	   --snapshot $snapdir  --snapshot-prefix test --weight $weights \
	   --units $units --lstm-depth $lstm_depth  --filters ${filters} #--lstm
fi
