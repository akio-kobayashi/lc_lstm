#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
cd /home/akio/lc_lstm

# librispeech
train=./train.h5
valid=./dev.h5
sorted=./train.sorted
valid_keys=./dev.sorted
test=./test.h5
n_labels=49
prior=./td_label_counts.h5

# features
feat_dim=40
units=160
lstm_depth=4
direction=bi
batch_size=16
factor=0.5
filters=16
learn_rate=1.0
optim=adadelta

# dir=model_d4_d160_f16_l1.0_B16_D0.0_f0.5_P3_LNtrue_BNtrue_vgg_adadelta_bi
snapdir=./model_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_B${batch_size}_Df${factor}_f${factor}_P3_LNtrue_BNtrue_vgg_${optim}_${direction}
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
