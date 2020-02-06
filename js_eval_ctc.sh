#!/bin/sh

device=$1
direction=$2
export CUDA_VISIBLE_DEVICES=$2
cd /home/akiokobayashi0809/js_lstm

# librispeech
train=./train_jnas.h5
valid=./dev_jnas.h5
keys=./train_jnas.sorted
valid_keys=./dev_jnas.sorted
n_labels=2363
prior=./js_label_counts.h5

# features
feat_dim=40
units=160
lstm_depth=5
batch_size=16
learn_rate=4.0e-4
factor=0.5
dropout=0.0
optim=adadelta

# js_logs_d5_d160_l4.0e-4_LNtrue_B16_D0.0_f0.5_P3_vgg_adadelta_uni
# models_d5_d160_l4.0e-4_LNtrue_B32_D0.0_f0.5_P3_vgg_adadelta_bi
snapdir=./model_d${lstm_depth}_d${units}_l${learn_rate}_LNtrue_B${batch_size}_D${dropout}_f${factor}_P3_vgg_${optim}_${direction}
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
fi
