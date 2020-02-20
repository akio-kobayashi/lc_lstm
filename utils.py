import numpy as np
import keras.utils

def expected_num_blocks(mat, procs, extras):
    length = mat.shape[0]

    if length < procs + extras:
        return [1, procs+extras]

    if (length - extras) % procs == 0:
        blocks = int((length-extras)/procs)
        frames = blocks * procs + extras
        return [blocks, frames]

    blocks = int((length - extras)/procs) + 1
    frames = blocks * procs + extras
    return [blocks, frames]

def split_utt(mat, procs, extras, n_blocks, feat_dim, max_blocks):
    '''
    return
        [ matrix w/ shape=[max_blocks, procs+extras, feat_dim],
            input_mask w/ shape=[max_blocks, procs+extras, feat_dim] ]
    '''
    length = mat.shape[0]
    src=np.zeros(shape=(max_blocks, procs+extras, feat_dim))
    mask=np.zeros(shape=(max_blocks, procs+extras, feat_dim))
    start = 0
    #true_length=[]
    for b in range(n_blocks):
        end = min([length, start+procs+extras])
        if length < start+procs+extras:
            frames = length-start
        else:
            frames = procs
            # or procs+extras in case all input-farmes are used
        mask[b, 0:frames, :] = 1.0
        src[b, 0:end-start, :] = np.expand_dims(mat[start:end, :], axis=0)
        #true_length.append(frames)
        if length < start+procs+extras:
            break
        start += procs
    return src, mask

def split_label(label, procs, extras, n_blocks, n_classes, max_blocks):
    '''
    return matrix w/ shape=[block_size, procs+extras, n_classes]
    '''
    length = len(label)
    src=np.zeros(shape=(max_blocks, procs+extras, n_classes))
    start = 0
    #true_length=[]
    for b in range(n_blocks):
        end = min([length, start+procs+extras])
        if length < start+procs+extras:
            frames = length - start
        else:
            frames = procs+extras
        labels = keras.utils.to_categorical(np.array(label[start:end]), num_classes=n_classes)
        src[b, 0:end-start:, :] = np.expand_dims(labels, axis=0)
        #true_length.append(frames)
        if length < start+procs+extras:
            break
        start += procs

    return src
