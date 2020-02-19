import numpy as np
import sys, os, re, gzip, struct
import random
import h5py
import copy
from keras import backend as K
from keras.utils import Sequence
import keras.utils
import tensorflow as tf
import utils

class FixedDataGenerator(Sequence):

    def __init__(self, file, key_file, batch_size=64, feat_dim=40, n_labels=1024,
        procs=10, extras=10, shuffle=False):

        self.file=file
        self.batch_size=batch_size
        self.feat_dim=feat_dim
        self.n_labels=n_labels
        self.procs=procs
        self.extras=extras
        self.shuffle=shuffle
        self.keys=[]
        self.sorted_keys=[]

        self.h5fd = h5py.File(self.file, 'r')
        self.n_samples = len(self.h5fd.keys())
        if key_file is not None:
            with open(key_file, 'r') as f:
                for line in f:
                    self.sorted_keys.append(line.strip())
        for key in self.h5fd.keys():
            self.keys.append(key)

        self.h5fd = h5py.File(self.file, 'r')
        self.n_samples = len(self.h5fd.keys())
        for key in self.h5fd.keys():
            self.keys.append(key)
        if len(self.sorted_keys) > 0:
            self.keys = self.sorted_keys

        #if self.shuffle:
        #    random.shuffle(self.keys)

    def __len__(self):
        return int(np.ceil(self.n_samples)/self.batch_size)
        #return int(np.floor(self.n_samples / self.batch_size))


    def __getitem__(self, index, return_keys=False):
        list_keys_temp = [self.keys[k] for k in range(index*self.batch_size,
                                                      min( (index+1)*self.batch_size,
                                                      len(self.keys) ) )]

        # [input_sequences, label_sequences, inputs_lengths, labels_length]
        x, mask, y
            =self.__data_generation(list_keys_temp)

        if return_keys is False:
            return [x, mask, y]
        else:
            return [x, mask, y], list_keys_temp

    def on_epoch_end(self):
        if self.shuffle == True:
            random.shuffle(self.keys)

    def __data_generation(self, list_keys_temp):

        max_num_blocks=0
        max_num_frames=0

        for i, key in enumerate(list_keys_temp):
            mat = self.h5fd[key+'/data'][()]
            [ex_blocks,ex_frames] = utils.expected_num_blocks(mat, self.procs, self.extras)
            if ex_blocks > max_num_blocks:
                max_num_blocks = ex_blocks
            if ex_frames > max_num_frames:
                max_num_frames = ex_frames

        input_mat=np.zeros(len(list_keys_temp), max_num_blocks, self.procs+self.extras, self.feat_dim)
        input_mask=np.zeros(len(list_keys_temp), max_num_blocks, self.procs+self.extras, self.feat_dim)
        output_labels=np.zeros(len(list_keys_temp), max_num_blocks, self.procs+self.extras, self.n_labels+1)

        for i, key in enumerate(list_keys_temp):
            mat = self.h5fd[key+'/data'][()]
            [ex_blocks, ex_frames] = utils.expected_num_blocks(mat, self.procs, self.extras)
            [blocked_mat, mask] = utils.split_utt(mat, self.procs, self.extras, ex_blocks,
                self.feat_dim, max_num_blocks)
            input_mat[i, :, :, :] = np.expand_dims(blocked_mat, axis=0)
            input_mask[i,:,:,:] = np.expand_dims(mask, axis=0)
            # label is a list of integers starting from 0
            label = self.h5fd[key+'/labels'][()]
            blocked_labels = utils.split_label(label, self.procs, self.extras, ex_blocks,
                self.n_labels+1, max_num_blocks)
            output_labels[i,:,:,:] = np.expand_dims(blocked_labels, axis=0)

        # transpose batch and block axes for outer loop in training
        input_mat = input_mat.transpose((1,0,2,3))
        input_mask = input_mask.transpose((1,0,2,3))
        output_labels = output_labels.transpose((1,0,2,3))

        return input_mat, input_mask, output_labels
