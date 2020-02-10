import numpy as np
import sys, os, re, gzip, struct
import random
import h5py
import copy
from keras import backend as K
from keras.utils import Sequence
import keras.utils
from keras.preprocessing import sequence
import tensorflow as tf

SORT_BLOCK_SIZE=256

class CEDataGenerator(Sequence):

    def __init__(self, file, key_file, batch_size=64, feat_dim=40, n_labels=1024, shuffle=False):

        self.file=file
        self.batch_size=batch_size
        self.feat_dim=feat_dim
        self.n_labels=n_labels
        self.shuffle=shuffle
        self.keys=[]
        self.starts=[]
        # input-length-ordered keys
        self.sorted_keys=[]

        self.h5fd = h5py.File(self.file, 'r')
        self.n_samples = len(self.h5fd.keys())
        if key_file is not None:
            with open(key_file, 'r') as f:
                for line in f:
                    self.sorted_keys.append(line.strip())
        for key in self.h5fd.keys():
            self.keys.append(key)

        #if len(self.sorted_keys) > 0:
        #    print(len(self.keys))
        #    print(len(self.sorted_keys))
        #    assert(len(self.keys) == len(self.sorted_keys))

        if self.shuffle:
            start=0
            while True:
                if start > len(self.sorted_keys):
                    break
                self.starts.append(start)
                start+=SORT_BLOCK_SIZE

            self.keys=[]
            start=0
            for n in range(len(self.starts)):
                start = self.starts[n]
                end = min(start+SORT_BLOCK_SIZE, len(self.sorted_keys))
                lst=self.sorted_keys[start:end]
                random.shuffle(lst)
                self.keys.extend(lst)
                start+=SORT_BLOCK_SIZE
            self.on_epoch_end()
        else:
            if len(self.sorted_keys) > 0:
                self.keys = self.sorted_keys

    def __len__(self):
        return int(np.ceil(self.n_samples)/self.batch_size)
        #return int(np.floor(self.n_samples / self.batch_size))


    def __getitem__(self, index, return_keys=False):
        list_keys_temp = [self.keys[k] for k in range(index*self.batch_size,
                                                      min( (index+1)*self.batch_size,
                                                      len(self.keys) ) )]

        # [input_sequences, label_sequences, inputs_lengths]
        # len(input_sequences) == len(label_sequences) = input_length
        input_sequences, label_sequences, masks, inputs_lengths = self.__data_generation(list_keys_temp)

        if return_keys is False:
            return [input_sequences, label_sequences, masks, inputs_lengths]
        else:
            return [input_sequences, label_sequences, masks,inputs_lengths], list_keys_temp

    def on_epoch_end(self):
        if self.shuffle == True:
            # shuffling start pointers
            random.shuffle(self.starts)
            self.keys=[]
            start=0
            for n in range(len(self.starts)):
                start = self.starts[n]
                end = min(start+SORT_BLOCK_SIZE, len(self.sorted_keys))
                lst=self.sorted_keys[start:end]
                random.shuffle(lst)
                self.keys.extend(lst)
                start+=SORT_BLOCK_SIZE

    def __data_generation(self, list_keys_temp):

        max_input_len=0
        max_output_len=0
        labels=[]
        in_seq=[]
        #mats=[]
        for i, key in enumerate(list_keys_temp):
            mat = self.h5fd[key+'/data'][()]
            label = self.h5fd[key+'/labels'][()]

            if mat.shape[0] > max_input_len:
              max_input_len = mat.shape[0]
            in_seq.append(mat.shape[0])

            # label is a list of integers starting from 0
            if len(label) > max_output_len:
                max_output_len = len(label)
            labels.append(label)
            #mats.append(mat)
            
        input_sequences = np.zeros((self.batch_size, max_input_len, self.feat_dim))
        label_sequences = np.zeros((self.batch_size, max_output_len, self.n_labels+1))
        masks = np.zeros((self.batch_size, max_output_len, 1))
        #print("%d %d" % (max_input_len, max_output_len))
        #print("%d %d" % (len(mats), len(labels)))
        '''
        if max_input_len == 0 or max_output_len == 0:
            for i in range(len(mats)):
                print(mats[i].shape)
                print(labels[i].shape)
        '''
        for i, key in enumerate(list_keys_temp):
            #mat = mats[i]
            mat = self.h5fd[key+'/data'][()]
            input_sequences[i, 0:mat.shape[0], :] = np.expand_dims(mat, axis=0)
            # lseq = (time, category)
            lseq = keras.utils.to_categorical(labels[i], num_classes=self.n_labels+1)
            label_sequences[i, 0:len(labels[i]),:] = np.expand_dims(lseq, axis=0)
            mask = np.ones((1, len(labels[i]), 1))
            masks[i, 0:len(labels[i]), :] = mask

        inputs_lengths=np.array(in_seq)

        return input_sequences, label_sequences, masks, inputs_lengths
