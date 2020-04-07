import numpy as np
import tensorflow as tf

def nbest_loss(acwt=0.6):

    def loss(y_true, y_pred):
        '''
        y_pred    : (block, batch, time, unit)
        y_true[0] : (block, batch, time, unit) # 1-best path
        y_true[1] : (block, batch) # log-scaled lm-score
        y_true[2] : (n-best, block, batch, time, unit) # n-best path
        y_true[3] : (n-best, block, batch) # log-scaled lm-scores
        '''

        # y_prd, y_true[0] must be > 0
        numer = tf.reduce_sum(tf.reduce_sum(tf.multiply(y_pred, y_true[0]),
                                            axis=-1), axis=-1)
        # (block, batch)
        numer = acwt * tf.log(numer)
        numer = tf.add(numer, y_true[1])

        denoms=[]
        for n in n_best:
            nb = tf.squeeze(y_true[2], axis=0)
            # (block, batch)
            denom = tf.reduce_sum(tf.reduce_sum(tf.multiply(y_pred, nb),
                                                axis=-1), axis=-1)
            denoms.append(tf.expand_dims(denom, axis=0))
        # (n-best, batch, block)
        denom = tf.concat(denoms, axis=0)
        denom = tf.add(tf.log(denom), y_true[3])

        # (batch, block)
        denom = tf.reduce_logsumexp(denom, axis=0)
        
        return -(numer - denom)

    return loss

