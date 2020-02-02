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
    parser.add_argument('--align', type=str, required=True, help='alignment data')
    parser.add_argument('--output', type=str, required=True, help='output file')

    args = parser.parse_args()

    with h5py.File(args.output, 'w') as output:
        with h5py.File(args.data, 'r') as data:
            keys = data.keys()
            with h5py.File(args.align, 'r') as align:
                for key in keys:
                    mat = data[key+'/data'][()]
                    seq = align[key+'/align'][()]
                    seq = seq.tolist()
                    output.create_group(key)
                    output.create_dataset(key+'/data', data=mat)
                    output.create_dataset(key+'/align', data=seq)

if __name__ == "__main__":
    main()
