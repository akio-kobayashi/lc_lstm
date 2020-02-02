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
    parser.add_argument('--keys', type=str, required=True, help='keys')
    args = parser.parse_args()

    keys=[]
    with open(args.keys, 'r') as f:
        for line in f:
            keys.append(line.strip())

    with h5py.File(args.data, 'r') as data:
        for key in keys:
            seq = data[key+'/labels'][()]
            seq = seq.tolist()
            print(key)
            print(seq)
if __name__ == "__main__":
    main()
