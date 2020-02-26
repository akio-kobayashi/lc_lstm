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

def pad_mat(mat, quad=False):
    if quad is False:
        if mat.shape[0]%2 == 1:
            temp = np.zeros((mat.shape[0]+1, mat.shape[1]))
            temp[0:mat.shape[0], :] = mat
            mat=temp
    else:
        mod = mat.shape[0]%4
        if mod > 0:
            temp = np.zeros((mat.shape[0] + (4-mod), mat.shape[1]))
            temp[0:mat.shape[0], :] = mat
            mat=temp
    return mat

