import numpy as np
import keras.utils
import ast
import tensorflow as tf

MAX_NBEST=10

def soft_loss(target, output):
    ''' y_true '''
    #output = output / tf.reduce_sum(output, axis=-1)
    #output = tf.clip_by_value(output, 1e-7, 1-1e-7)
    loss = -tf.reduce_sum(target * tf.math.log(output), axis=-1)
    #loss = -tf.reduce_mean(tf.multiply(y_true, tf.log(y_pred)), axis=-1)

    return loss

def soft_acc(target, output):
    ''' y_true: (batch, time, feats) '''
    error = tf.reduce_sum(np.multiply(1.0-target, output), axis=-1)
    accuracy = 1.0 - error

    return accuracy

def str2nbest(strlist):
    nbest=[]
    for n in range(strlist):
        labels = str2dict(strlist[n])
        nbest.append(labels)
    return nbest

def str2dict(posts):
    post_labels=[]

    for n in range(len(posts)):
        #d = ast.literal_eval(posts[n])
        d=eval(posts[n])
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
    length = len(post_labels)
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
            for k in range(frames):
                d = post_labels[start+k]
                for key in d:
                    src[num_blocks, k, key] = d[key]
        else:
            end = start + procs + extras2
            num_frames += procs + extras2
            if length < end:
                frames = length-start
            else:
                frames = procs + extras2
            for k in range(frames):
                d = post_labels[start+k]
                for key in d:
                    src[num_blocks, k, key] = d[key]
        start += procs
        num_blocks+=1

    return src

def split_nbest_label(nbest_labels, procs, extras1, extras2, num_extras1, n_blocks, n_classes, max_blocks):
    '''
    post_labels = list of (list of dict)
    return matrix w/ shape=[nbest, block_size, procs+extras, n_classes]
    '''

    nbest=len(nbest_labels)
    length = len(nbest_labels[0])
    
    max_extras = max([extras1, extras2])
    src=np.zeros(shape=(MAX_NBEST, max_blocks, procs+max_extras, n_classes))

    for n in range(nbest):    
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
                for k in range(frames):
                    d = post_labels[start+k]
                    for key in d:
                        src[n, num_blocks, k, key] = d[key]
            else:
                end = start + procs + extras2
                num_frames += procs + extras2
                if length < end:
                    frames = length-start
                else:
                    frames = procs + extras2
                for k in range(frames):
                    d = post_labels[start+k]
                    for key in d:
                        src[n, num_blocks, k, key] = d[key]
            start += procs
            num_blocks+=1

    return src
