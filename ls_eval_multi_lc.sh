#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

# librispeech
valid=./dev_clean.h5
test=./test_clean.h5
n_labels=32
prior=./ls_label_counts.h5

# features
feat_dim=40
units=256
lstm_depth=4
direction=bi
batch_size=256
learn_rate=1.0
factor=0.9
dropout=0.0
optim=adadelta
filters=32

proc_frames=10
extra_frames1=10
extra_frames2=10
num_procs=1

snapdir=./ls/lc_model_d${lstm_depth}_d${units}_f${filters}_l${learn_rate}
snapdir=${snapdir}_B${batch_size}_D${dropout}
snapdir=${snapdir}_f${factor}_LNtrue_vgg_lstm
snapdir=${snapdir}_p${proc_frames}_ef${extra_frames1}_es${extra_frames2}
snapdir=${snapdir}_n${num_procs}
snapdir=${snapdir}_${optim}_${direction}

weights=${snapdir}/snapshot.h5

if [ -e $weights ]; then
    python multi_lc_eval_lstm.py --data $valid \
	   --prior $prior \
	   --feat-dim $feat_dim --n-labels $n_labels \
	   --snapshot $snapdir  --snapshot-prefix lc_dev \
	   --weight $weights --lstm \
	   --units $units --lstm-depth $lstm_depth --filters $filters \
	   --process-frames $proc_frames --extra-frames1 $extra_frames1 \
	   --extra-frames2 $extra_frames2 --num-extra-frames $num_procs

    python multi_lc_eval_lstm.py --data $test --prior $prior \
	   --feat-dim $feat_dim --n-labels $n_labels \
	   --snapshot $snapdir  --snapshot-prefix ce_test \
	   --weight $weights --filters $filters --lstm \
	   --units $units --lstm-depth $lstm_depth \
	   --process-frames $proc_frames --extra-frames1 $extra_frames1 \
	   --extra-frames2 $extra_frames2 --num-extra-frames $num_procs
fi
