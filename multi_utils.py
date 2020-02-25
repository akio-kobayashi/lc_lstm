import numpy as np
import keras.utils

def expected_num_blocks(mat, procs, extras1, extras2, num_extras1=1):
    length = mat.shape[0]

    if length < procs + extras1:
        return [1, procs+extras1]

    start = 0
    num_blocks = 0
    num_frames = 0
    while True:
        if start > length:
            break
        if num_blocks < num_extras1:
            end = start + procs + extras1
            num_frames += procs + extras1
            if length < end:
                num_blocks += 1
                break
        else:
            end = start + procs + extras2
            num_frames += procs + extras2
            if length < end:
                num_blocks += 1
                break
        start += procs
        num_blocks += 1

    return [num_blocks, num_frames]

def split_utt(mat, procs, extras1, extras2, num_extras1, n_blocks, feat_dim, max_blocks):
    '''
    return
        [ matrix w/ shape=[max_blocks, procs+extras, feat_dim],
            input_mask w/ shape=[max_blocks, procs+extras, feat_dim] ]
    '''
    length = mat.shape[0]
    max_extras = max([extras1, extras2])
    src=np.zeros(shape=(max_blocks, procs+max_extras, feat_dim))
    mask=np.zeros(shape=(max_blocks, procs+max_extras, feat_dim))

    start = 0
    num_blocks = 0
    num_frames = 0
    while True:
        if start > length:
            break
        if num_blocks < num_extras1:
            end = start + procs + extras1
            num_frames += procs + extras1
            if length < end:
                mask_frames = length - start
                frames = mask_frames
            else:
                mask_frames = procs
                frames = procs + extras1
            mask[num_blocks, 0:mask_frames, :] = 1.0
            src[num_blocks, 0:frames, :] = np.expand_dims(mat[start:start+frames, :], axis=0)
        else:
            end = start + procs + extras2
            num_frames += procs + extras2
            if length < end:
                mask_frames = length - start
                frames = mask_frames
            else:
                mask_frames = procs
                frames = procs + extras2
            mask[num_blocks, 0:frames, :] = 1.0
            src[num_blocks, 0:frames, :] = np.expand_dims(mat[start:start+frames, :], axis=0)

        start += procs
        num_blocks += 1

    return src, mask

def split_label(label, procs, extras1, extras2, num_extras1, n_blocks, n_classes, max_blocks):
    '''
    return matrix w/ shape=[block_size, procs+extras, n_classes]
    '''
    length = len(label)
    max_extras = max([extras1, extras2])
    src=np.zeros(shape=(max_blocks, procs+max_extras, n_classes))
    #mask=np.zeros(shape=(max_blocks, procs+max_extras, n_classes))
    start = 0
    num_blocks = 0
    num_frames = 0

    while True:
        if start > length:
            break
        if num_blocks < num_extras1:
            end = start + procs + extras1
            num_frames += procs + extras1
            if length < end:
                frames = length - start
                #mask_frames = frames
            else:
                frames = procs + extras1
                #mask_frames = procs + extras1
            labels = keras.utils.to_categorical(np.array(label[start:end]), num_classes=n_classes)
            src[num_blocks, 0:frames, :] = np.expand_dims(labels, axis=0)
            #mask[num_blocks, 0:mask_frames,:]=1.0

        else:
            end = start + procs + extras2
            num_frames += procs + extras2
            if length < end:
                frames = length-start
                #mask_frames = frames
            else:
                frames = procs + extras2
                #mask_frames = procs + extras2
            labels = keras.utils.to_categorical(np.array(label[start:end]), num_classes=n_classes)
            src[num_blocks, 0:frames, :] = np.expand_dims(labels, axis=0)
            #mask[num_blocks, 0:frames,:]=1.0
        start += procs
        num_blocks+=1

    return src
    #, mask
