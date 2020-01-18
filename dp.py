import numpy as np

def dynamic_programming(scores, labels, n_inputs, n_labels):
    '''
    params:
        scores: 2-d np.array with shape=(frames, n_labels), log-scores
        labels: 1-d np.array w/o blanks
    return: 1-d np.array
    '''
    seqlen = 2*labels.shape[0]+1

    labels_blanks = np.full(n_labels)
    for n in range(labels.shape[0]):
        labels_blanks[2*n+1] = labels[n]

    dpmat = np.full((n_inputs, seqlen), -1.0e10)
    bptr  = np.full((n_inputs, seqlen), -1)
    # init
    dpmat[0][0] = scores[0][n_labels]
    dpmat[0][1] = scores[0][labels_blanks[1]]

    for f in range(n_inputs):
        if f == 0 continue
        for curr_state in (seqlen):
            if labels_blanks[curr_state]==n_labels:
                prev_state = min(0, curr_state-1)
            else:
                prev_state = min(0, curr_state-2)
            min_score=-1.0e10
            min_state=-1
            while p <= curr_state:
                score = dpmat[f-1][p] + dpmat[f][curr_state]
                if min_score < score:
                    min_score = score
                    min_state = p
                p+=1
            dpmat[f][curr_state] = min_score
            bptr[f][curr_state] = p
    final_state = seqlen-1
    if dpmat[n_inputs-1][final_state-1] > dpmat[n_inputs-1][final_state-2]:
        final_state = final_state-1

    results=[]
    state = final_state
    fpt=n_inputs-1
    while state >= 0:
        results.append(state)
        state=bptr[fpt][state]
        fpt -= 1
    results = results[::-1]

    return results
