#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
cd /home/akio/lc_lstm

# librispeech
train=./train_clean_100.h5
valid=./dev_clean.h5
sorted=./train_clean_100.sorted
test=./test_clean.h5
n_labels=32
prior=./ls_label_counts.h5

# features
feat_dim=40
units=160
lstm_depth=5
#training
#direction=bi
batch_size=32
learn_rate=4.0e-4
factor=0.5
direction=bi

snapdir=./model_d${lstm_depth}_d${units}_l${learn_rate}_B${batch_size}_D_f${factor}_P3_LNtrue_vgg_adadelta_${direction}
weights=${snapdir}/snapshot.h5

if [ -e $weights ]; then
    python eval_ctc_lstm.py --data $train --key-file $sorted --prior $prior --direction bi \
	   --feat-dim $feat_dim --n-labels $n_labels --vgg true \
	   --snapshot $snapdir  --weight $weights \
	   --units $units --lstm-depth $lstm_depth --align true
fi
