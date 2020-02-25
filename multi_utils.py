import numpy as np

def multi_expected_num_blocks(mat, procs, extras1, extras2, num_extras1=1):
    length = mat.shape[0]

    if length < procs + extras1
        return [1, procs+extras]

    start = 0
    num_blocks = 0
    num_frames = 0
    while True:
        if start > length:
            break
        if num_blocks < num_extras1:
            end = start + proc + extras1
            nun_blocks += 1
            num_frames += proc + extras1
            if length < end:
                break
        else:
            end = start + proc + extras2
            num_blocks += 1
            num_frames += proc + extras2
            if length < end:
                break
        start += procs

    return [num_blocks, num_frames]

def multi_split_utt(mat, procs, extras1, extras2, num_extras1, n_blocks, feat_dim, max_blocks):
    '''
    return
        [ matrix w/ shape=[max_blocks, procs+extras, feat_dim],
            input_mask w/ shape=[max_blocks, procs+extras, feat_dim] ]
    '''
    length = mat.shape[0]
    max_extras = max([extras1, extras2])
    src=np.zeros(shape=(max_blocks, procs+max_extras, feat_dim))
    mask=np.zeros(shape=(max_blocks, proc+max_extras, feat_dim))

    start = 0
    num_blocks = 0
    num_frames = 0
    while True:
        if start > length:
            break
        if num_blocks < num_extras1:
            end = start + procs + extras1
            num_blocks += 1
            num_frames += procs + extras1
            if length < end:
                mask_frames = length - start
                frames = mask_frames
            else:
                mask_frames = procs
                frames = procs + extra1
            mask[b, 0:mask_frames, :] = 1.0
            src[b, 0:frames, :] = np.expand_dims(mat[start:start+frames, :], axis=0)
        else:
            end = start + proc + extras2:
            num_blocks += 1
            num_frames += procs + extras2
            if length < end:
                mask_frames = length - start
                frames = mask_frames
            else:
                mask_frames = procs
                frames = procs + extra2
            mask[b, 0:framds, :] = 1.0
            src[b, 0:frames, :] = np.expand_dims(mat[start:start+frames, :], axis=0)

        start += procs

    return src, mask

def split_label(label, procs, extras1, extras2, num_extras1, n_blocks, n_classes, max_blocks):
    '''
    return matrix w/ shape=[block_size, procs+extras, n_classes]
    '''
    length = len(label)
    max_extras = max([extras1, extras2])
    src=np.zeros(shape=(max_blocks, proc+max_extras, n_classes))
    start = 0
    num_blocks = 0
    num_frames = 0

    while True:
        if start > length:
            break
        if num_blocks < num_extras1:
            end = start + procs + extras1
            num_blocks += 1
            num_frames += procs + extras1
            if length < end:
                frames = length - start
            else:
                frames = procs
            labels = keras.utils.to_categorical(np.array(label[start:end]), num_classes=n_classes)
            src[b, 0:frames, :] = np.expand_dims(labels, axis=0)
        else:
            end = start + procs + extras2
            num_blocks += 1
            num_frames += procs + extras2
            if length < end:
                frames = length-start
            else:
                frames = procs
            labels = keras.utils.to_categorical(np.array(label[start:end]), num_classes=n_classes)
            src[b, 0:frames, :] = np.expand_dims(labels, axis=0)
        start += procs

    return src, mask
