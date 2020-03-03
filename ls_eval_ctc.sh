#!/bin/sh
direction=bi

#cd /home/akio/lc_lstm
train=./train_clean_100.h5
valid=./dev_clean.h5
test=./test_clean.h5
keys=./train_clean_100.sorted
valid_keys=./dev_clean.sorted
export CUDA_VISIBLE_DEVICES=0
n_labels=32
prior=./ls_label_counts.h5

# features
feat_dim=40
units=256
lstm_depth=4
batch_size=16
learn_rate=1.0
factor=0.9
dropout=0.0
optim=adadelta
filters=32
epochs=100

# model_d4_d256_f32_l1.0_ml1.0e-2_LNtrue_BNtrue_B16_D0.0_f0.9_vgg_lstm_adadelta_ep100_bi
snapdir=./ls/model_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}_ml1.0e-2_LNtrue_BNtrue_B${batch_size}_D${dropout}_f${factor}_vgg_lstm_${optim}_ep${epochs}_${direction}
echo $snapdir
weights=${snapdir}/snapshot.h5

if [ -e $weights ]; then
    python eval_ctc_lstm.py --data $train --key-file $keys --direction ${direction} \
	   --feat-dim $feat_dim --n-labels $n_labels \
	   --snapshot $snapdir  --snapshot-prefix train --weight $weights \
	   --units $units --lstm-depth $lstm_depth  --filters ${filters} --lstm

    python eval_ctc_lstm.py --data $valid --key-file $valid_keys --direction ${direction} \
	   --feat-dim $feat_dim --n-labels $n_labels \
	   --snapshot $snapdir  --snapshot-prefix dev --weight $weights \
	   --units $units --lstm-depth $lstm_depth  --filters ${filters} --lstm

    python eval_ctc_lstm.py --data $test --direction ${direction} \
	   --feat-dim $feat_dim --n-labels $n_labels \
	   --snapshot $snapdir  --snapshot-prefix test --weight $weights \
	   --units $units --lstm-depth $lstm_depth  --filters ${filters} --lstm
fi
