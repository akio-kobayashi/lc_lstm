
def expected_num_blocks(mat, procs, extras):
    length = mat.shape[0]

    if length < procs + extras
        return [1, procs+extras]

    if (length - extras) % procs == 0:
        blocks = (length-extras)/procs
        frames = blocks * procs + extras
        return [blocks, frames]

    blocks = int((length - extras)/procs) + 1
    frames = blocks * procs + extras
    return [blocks, frames]

def split_utt(mat, procs, extras, n_blocks, feat_dim):
    '''
    return
        [ matrix w/ (shape=[n_blocks, time, feat]), input_length]
    '''
    length = mat.shape[0]
    block_length=[]

    src=np.array(shape=(n_blocks, procs+extras, feat_dim))
    start = 0
    for b in range(n_blocks):
        end = np.min(length, start+procs+extras)
        if length < start+procs+extras:
            frames = length
        else:
            frames = procs
            # or procs+extras in case all input-farmes are used
        block_length.append(frames)
        src[b, frames, :] = np.expand_dims(mat[start:end, :], axis=0)
        start += procs
    input_length = np.array(block_length)

    return [src, input_length]

def split_label(label, procs, extras, n_blocks):
    '''
    return [ matrix w/ (block_size, max_frames), label_length ]
    '''
    block_list=[]
    length = len(label)
    start = 0
    max_frames=0
    for b in range(n_blocks):
        end = np.min(length, start+procs+extras)
        if length < start+procs+extras:
            frames = length
        else:
            frames = procs
        if frames > max_frames:
            max_frames = frames
        block_length.append(frames)
        lst = label[start:end]
        block_list.append(lst)
    blocks = pad_sequences(block_list, maxlen=max_frames, padding='post', value=0.0)
    label_length = np.array(block_list)

    return [blocks, label_length]
