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
        for key in data.keys():
            mat = data[key+'/data'][()]
            lbl = data[key+'/labels'][()]
            if mat.shape[0] != lbl.shape[0]:
                continue
            elif mat.shape[0] == 0 and lbl.shape[0] == 0:
                continue
            else:
                print ("%s %d %d" % (key, mat.shape[0], lbl.shape[0]))
                
if __name__ == "__main__":
    main()
