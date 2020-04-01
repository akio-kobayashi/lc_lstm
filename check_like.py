import argparse
import os
import sys
import subprocess
import time
import numpy as np
import random
import h5py

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='training data')
    args = parser.parse_args()

    with h5py.File(args.data, 'r') as data:
        keys = data.keys()
        for key in keys:
            mat = data[key+'/likelihood'][()]
            mat = np.roll(mat, shift=1, axis=-1)
            #if np.isinf(mat).any():
            #if np.isnan(mat).any():
            print(key)
            label = np.argmax(mat, axis=-1)
            print(label)
           
if __name__ == "__main__":
    main()
