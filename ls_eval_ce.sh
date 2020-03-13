#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
#cd /home/akio/lc_lstm

# librispeech
#train=./train_clean_100.h5
valid=./dev_clean.h5
#sorted=./train_clean_100.sorted
valid_keys=./dev_clean.sorted
test=./test_clean.h5
n_labels=32
prior=./ls_label_counts.h5

# features
feat_dim=40
units=256
lstm_depth=4
direction=bi
batch_size=16
learn_rate=1.0
factor=0.9
dropout=0.0
optim=adadelta
filters=32

snapdir=./ls/ce_model_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}
snapdir=${snapdir}_LNtrue_BNtrue_B${batch_size}_D${dropout}
snapdir=${snapdir}_f${factor}_vgg_lstm_${optim}_${direction}
weights=${snapdir}/snapshot.h5

if [ -e $weights ]; then
    python eval_ce_lstm.py --data $valid --key-file $valid_keys \
	   --prior $prior --direction ${direction} \
	   --feat-dim $feat_dim --n-labels $n_labels \
	   --snapshot $snapdir  --snapshot-prefix ce_dev \
	   --weight $weights --lstm \
	   --units $units --lstm-depth $lstm_depth --filters $filters

    python eval_ce_lstm.py --data $test --prior $prior \
	   --direction ${direction} \
	   --feat-dim $feat_dim --n-labels $n_labels \
	   --snapshot $snapdir  --snapshot-prefix ce_test \
	   --weight $weights --filters $filters --lstm \
	   --units $units --lstm-depth $lstm_depth
fi
