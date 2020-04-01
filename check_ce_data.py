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
    parser.add_argument('--num', type=int)
    args = parser.parse_args()

    with h5py.File(args.data, 'r') as data:
        for key in data.keys():
            mat = data[key+'/data'][()]
            lbl = data[key+'/labels'][()]
            #if mat.shape[0] != lbl.shape[0]:
            #    continue
            #elif mat.shape[0] == 0 and lbl.shape[0] == 0:
            #    continue
            #else:
            #    print ("%s %d %d" % (key, mat.shape[0], lbl.shape[0]))
            print(key)
            lbl += 1
            lbl = np.where(lbl>args.num, 0, lbl)
            print(lbl)
            
if __name__ == "__main__":
    main()
