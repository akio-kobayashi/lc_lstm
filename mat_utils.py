import numpy as np
import sys, os, re, gzip, struct
import random
import h5py
import copy
import keras.backend as K
from keras.utils import Sequence
import keras.utils
from keras.preprocessing import sequence
import tensorflow as tf

def pad_mat(mat, mod=1):
    m = mat.shape[0]%mod
    temp = np.zeros((mat.shape[0]+(mod-m), mat.shape[1]))
    temp[0:mat.shape[0],:] = mat

    return mat

def pad_label(label, mod=1):
    m = len(label)%mod
    temp = label
    for k in range(mod-m):
        temp.append(label[-1])
    return temp
