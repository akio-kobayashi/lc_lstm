#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
#cd /home/akio/lc_lstm

# librispeech
valid=./dev.h5
test=./test.h5
prior=./td_label_counts.h5
n_labels=49

# features
feat_dim=40
units=256
lstm_depth=4
direction=bi
batch_size=16
factor=0.9
filters=32
learn_rate=1.0
optim=adadelta
dropout=0.0

snapdir=./td/ce_model_d${lstm_depth}_d${units}
snapdir=${snapdir}_l${learn_rate}_B${batch_size}_D${dropout}_f${factor}
snapdir=${snapdir}_LNtrue_vgg_lstm_${optim}_${direction}
echo $snapdir

weights=${snapdir}/snapshot.h5

if [ -e $weights ]; then
    python eval_ce_lstm.py --data $valid --prior $prior --direction ${direction} \
	   --feat-dim $feat_dim --n-labels $n_labels --filters $filters \
	   --snapshot $snapdir  --snapshot-prefix ce_dev --weight $weights \
	   --units $units --lstm-depth $lstm_depth --lstm

    python eval_ce_lstm.py --data $test --prior $prior \
	   --direction ${direction} --filters ${filters} \
	   --feat-dim $feat_dim --n-labels $n_labels \
	   --snapshot $snapdir  --snapshot-prefix ce_test --weight $weights \
	   --units $units --lstm-depth $lstm_depth --lstm
fi
