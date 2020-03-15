import numpy as np
import keras.utils
import ast

def soft_loss(y_true, y_pred):
    ''' y_true '''

    loss = np.reduce_mean(np.multiply(y_true, np.log(y_pred)), axis=-1)

    return loss

def soft_acc(y_true, y_pred):
    ''' y_true: (batch, time, feats) '''

    hots = np.where(y_true > 0.0).astype(np.float)
    numer = np.sum(np.sum(np.multiply(hots, y_pred), axis=-1), axis=-1)
    # denom = (batch,)
    denom = np.sum(np.where(np.sum(np.where(y_true > 0.0).astype(np.int), axis=-1) > 0.0).astype(float), axis=-1)

    return np.divide(numer, denom)

def str2dict(posts):
    post_labels=[]

    for n in range(len(posts)):
        d = ast.literal_eval(posts[n])
        for k in d:
            d[k] = float(d[k])
        post_labels.append(d)

    return post_labels

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
    mask=np.zeros(shape=(max_blocks, procs+max_extras, 1))
    label_mask = np.zeros(shape=(max_blocks, procs+max_extras, 1))

    start = 0
    num_blocks = 0
    num_frames = 0
    while num_blocks < max_blocks:
        if start > length:
            break
        if num_blocks < num_extras1:
            end = start + procs + extras1
            num_frames += procs + extras1
            if length < end:
                mask_frames = length-start
                frames = mask_frames
                mask_label_frames = mask_frames
            else:
                mask_frames = procs+extras1
                frames = procs + extras1
                mask_label_frames = procs
            mask[num_blocks, 0:mask_frames, :] = 1.0
            src[num_blocks, 0:frames, :] = np.expand_dims(mat[start:start+frames, :], axis=0)
            label_mask[num_blocks, 0:mask_label_frames, :] = 1.0
        else:
            end = start + procs + extras2
            num_frames += procs + extras2
            if length < end:
                mask_frames = length-start
                frames = mask_frames
                mask_label_frames = mask_frames
            else:
                mask_frames = procs
                frames = procs + extras2
                mask_label_frames = procs
            mask[num_blocks, 0:frames, :] = 1.0
            src[num_blocks, 0:frames, :] = np.expand_dims(mat[start:start+frames, :], axis=0)
            label_mask[num_blocks, 0:mask_label_frames, :] = 1.0

        start += procs
        num_blocks += 1

    return src, mask, label_mask

def split_label(label, procs, extras1, extras2, num_extras1, n_blocks, n_classes, max_blocks):
    '''
    return matrix w/ shape=[block_size, procs+extras, n_classes]
    '''
    length = len(label)
    max_extras = max([extras1, extras2])
    src=np.zeros(shape=(max_blocks, procs+max_extras, n_classes))
    start = 0
    num_blocks = 0
    num_frames = 0

    while num_blocks < max_blocks:
        if start > length:
            break
        if num_blocks < num_extras1:
            end = start + procs + extras1
            num_frames += procs + extras1
            if length < end:
                frames = length-start
            else:
                frames = procs+extras1
            labels = keras.utils.to_categorical(np.array(label[start:end]), num_classes=n_classes)
            src[num_blocks, 0:frames, :] = np.expand_dims(labels, axis=0)
        else:
            end = start + procs + extras2
            num_frames += procs + extras2
            if length < end:
                frames = length-start
            else:
                frames = procs + extras2
            labels = keras.utils.to_categorical(np.array(label[start:end]), num_classes=n_classes)
            src[num_blocks, 0:frames, :] = np.expand_dims(labels, axis=0)
        start += procs
        num_blocks+=1

    return src

def split_post_label(post_labels, procs, extras1, extras2, num_extras1, n_blocks, n_classes, max_blocks):
    '''
    post_labels = list including dict
    return matrix w/ shape=[block_size, procs+extras, n_classes]
    '''
    length = len(label)
    max_extras = max([extras1, extras2])
    src=np.zeros(shape=(max_blocks, procs+max_extras, n_classes))
    start = 0
    num_blocks = 0
    num_frames = 0

    while num_blocks < max_blocks:
        if start > length:
            break
        if num_blocks < num_extras1:
            end = start + procs + extras1
            num_frames += procs + extras1
            if length < end:
                frames = length-start
            else:
                frames = procs+extras1
            for k in range(len(end - start)):
                d = post_labels[start+k]
                for key, val in d:
                    src[num_blocks, k, key] = val
        else:
            end = start + procs + extras2
            num_frames += procs + extras2
            if length < end:
                frames = length-start
            else:
                frames = procs + extras2
            for k in range(len(end - start)):
                d = post_labels[start+k]
                for key, val in d:
                    src[num_blocks, k, key] = val
        start += procs
        num_blocks+=1

    return src
