import numpy as np
import sys, os, re, gzip, struct
import random
import h5py
import copy
from keras import backend as K
from keras.utils import Sequence
import keras.utils
import tensorflow as tf

class DataGenerator(Sequence):

    def __init__(self, file, key_file, batch_size=64, feat_dim=40, n_labels=1024, shuffle=False):

        self.file=file
        self.batch_size=batch_size
        self.feat_dim=feat_dim
        self.n_labels=n_labels
        self.shuffle=shuffle
        self.keys=[]

        self.h5fd = h5py.File(self.file, 'r')
        self.n_samples = len(self.h5fd.keys())
        if key_file is not None:
            with open(key_file, 'r') as f:
                for line in f:
                    self.keys.append(f.strip())
        else:
            for key in self.h5fd.keys():
                self.keys.append(key)

        if self.shuffle:
            random.shuffle(self.keys)

    def __len__(self):
        return int(np.ceil(self.n_samples)/self.batch_size)
        #return int(np.floor(self.n_samples / self.batch_size))


    def __getitem__(self, index, return_keys=False):
        list_keys_temp = [self.keys[k] for k in range(index*self.batch_size,
                                                      min( (index+1)*self.batch_size,
                                                      len(self.keys) ) )]

        # [input_sequences, label_sequences, inputs_lengths, labels_length]
        input_sequences, label_sequences, inputs_lengths, labels_lengths = self.__data_generation(list_keys_temp)

        if return_keys is False:
            return [input_sequences, label_sequences, inputs_lengths, labels_lengths]
        else:
            return [input_sequences, label_sequences, inputs_lengths, labels_lengths], list_keys_temp

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.keys)

    def __data_generation(self, list_keys_temp):

        max_input_len=0
        max_output_len=0
        labels=[]
        for i, key in enumerate(list_keys_temp):
            mat = self.h5fd[key+'/data'][()]
            if mat.shape[0] > max_len:
              max_input_len = mat.shape[0]
            in_seq.append(mat.shape[0])

            # label is a list of integers starting from 0
            label = self.h5fd[key+'/labels'][()]
            labels.append(label)
            if len(ids) > max_output_len:
              max_output_len = len(ids)
            lb_seq.append(len(ids))

        input_sequences = np.zeros((self.batch_size, max_input_len, self.feat_dim))
        for i, key in enumerate(list_keys_temp):
            mat = self.h5fd([key])[()]
            input_sequences[i, 0:mat.shape[0], :] = np.expand_dims(mat, axis=0)

        label_sequences=pad_sequences(lb_seq, maxlen=max_output_len, padding='post', value=0)
        inputs_lengths=np.arrray(in_seq)
        labels_lengths=np.array(lb_seq)

        return input_sequences, label_sequences, inputs_lengths, labels_lengths
