import numpy as np

def remove_duplicates(ary):
    #a[(a*np.sign(abs(np.diff(np.concatenate(([0], a)))))).nonzero()]
    ary = ary * np.sign(abs(np.diff(np.concatenate(([0], ary)))))
    ary = ary.nonzero()

    return ary

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, required=True, help='training data')
    parser.add_argument('--syms', type=str, required=True, help='symbols')
    args = parser.parse_args()

    with open(args.syms, 'r') as file:
        lines = file.readlines()
        lines = [l.strip() for l in lines]
        for l in lines:
            item = l.split()
            int2sym[int(item[1])] = item[0]

    with h5py.File(args.data, 'r') as data:
        keys = data.keys()

        for key in keys:

            mat = data[key+'/likelihood'][()]
            mat = np.roll(mat, shift=1, axis=-1)
            label = np.argmax(mat, axis=-1)
            label = remove_duplicates(label)
            words=[]
            for id in label:
                if id > 0:
                    words.append(int2sym[id])

            words = ' '.join(words)
            print("%s %s" % (key, words))

if __name__ == "__main__":
    main()
