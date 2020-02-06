#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
cd /home/akio/lc_lstm

# librispeech
train=./train_clean_100.h5
valid=./dev_clean.h5
sorted=./train_clean_100.sorted
valid_keys=./dev_clean.sorted
test=./test_clean.h5
n_labels=32
prior=./ls_label_counts.h5

# features
feat_dim=40
units=160
lstm_depth=5
direction=bi
batch_size=32
learn_rate=4.0e-4
factor=0.5
dropout=0.0

# logs_d5_d160_l4.0e-4_LNtrue_B32_D0.0_f0.5_P3_vgg_adadelta_bi/
snapdir=./model_d${lstm_depth}_d${units}_l${learn_rate}_LNtrue_B${batch_size}_D${dropout}_f${factor}_P3_vgg_${direction}
weights=${snapdir}/snapshot.h5

if [ -e $weights ]; then
    python eval_ctc_lstm.py --data $train --key-file $sorted --prior $prior --direction ${direction} \
	   --feat-dim $feat_dim --n-labels $n_labels \
	   --snapshot $snapdir  --snapshot-prefix train --weight $weights \
	   --units $units --lstm-depth $lstm_depth

    python eval_ctc_lstm.py --data $valid --key-file $valid_keys --prior $prior --direction ${direction} \
	   --feat-dim $feat_dim --n-labels $n_labels \
	   --snapshot $snapdir  --snapshot-prefix dev --weight $weights \
	   --units $units --lstm-depth $lstm_depth

    python eval_ctc_lstm.py --data $test --prior $prior --direction ${direction} \
	   --feat-dim $feat_dim --n-labels $n_labels \
	   --snapshot $snapdir  --snapshot-prefix test --weight $weights \
	   --units $units --lstm-depth $lstm_depth
fi
