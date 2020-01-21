import numpy as np
#import dynamic_programming

def dynamic_programming(scores, labels, n_inputs, n_labels):
    '''
    params:
        scores: 2-d np.array with shape=(frames, n_labels+1), log-scores
        labels: 1-d np.array w/o blanks
    return: 1-d np.array
    '''

    #print(labels.shape)
    blank=0
    #print(np.argmax(scores, axis=-1))
    seqlen = 2*labels.shape[0]+1

    #print(labels.shape)
    labels_blanks = np.full((seqlen, 1), blank) # filled with blanks
    for n in range(labels.shape[0]):
        labels_blanks[2*n+1] = labels[n]

    dpmat = np.full((n_inputs, seqlen), -1.0e10)
    bptr  = np.full((n_inputs, seqlen), -1)
    #print(bptr.shape)
    # init
    dpmat[0][0] = scores[0][blank]
    dpmat[0][1] = scores[0][labels_blanks[1]]
    #print('%f %f' % (dpmat[0][0], dpmat[0][1]))
    for f in range(n_inputs):
        if f == 0: continue
        for curr_state in range(seqlen):
            if labels_blanks[curr_state]==blank:
                prev_state = max(0, curr_state-1)
            else:
                prev_state = max(0, curr_state-1)
            max_score=-1.0e10
            max_state=-1
            p=prev_state
            while p <= curr_state:
                score = dpmat[f-1][p] + scores[f][labels_blanks[curr_state]]
                #print('%d %d %d %f' % (f, labels_blanks[p], labels_blanks[curr_state], scores[f][labels_blanks[curr_state]]))
                if max_score < score:
                    max_score = score
                    max_state = p
                p+=1
            #print('%d %d %d %f' % (f, min_state, curr_state, min_score))
            dpmat[f][curr_state] = max_score
            bptr[f][curr_state] = max_state
            #print('%d %d %f' % (labels_blanks[max_state], labels_blanks[curr_state], max_score))
    final_state = seqlen-1
    if dpmat[n_inputs-1][final_state-1] < dpmat[n_inputs-1][final_state-2]:
        final_state = final_state-2

    results=[]
    state = final_state
    fpt=n_inputs-1
    while state >= 0:
        results.append(labels_blanks[state,0])
        state=bptr[fpt][state]
        fpt -= 1
    results = results[::-1]
    print(results)
    
    # results shpae=(input_length, )
    return np.array(results)
