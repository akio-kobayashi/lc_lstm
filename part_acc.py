import keras.utils
import keras.backend as K
import numpy as np
import tensorflow as tf

def part_loss_acc(y_true, y_pred, label_mask):
    """ y_true : (batch, time, num_classes)
        y_pred : 
        mask   : (batch, time, num_classes)
    """

    """ loss """
    loss = np.multiply(y_true, np.log(y_pred))
    loss = np.multiply(loss, label_mask)
    loss = np.divide(np.sum(loss, axis=-1), np.sum(label_mask, axis=-1))
    
    """ accuracy """
    correct = K.cast(K.equal(K.argmax(y_true, axis=-1),
                             K.argmax(y_pred, axis=-1),
                             K.floatx()))
    
    correct = np.multiply(correct, label_mask)
    correct = np.sum(correct, axis=-1)
    acc = np.divide(correct, np.sum(label_mask, axis=-1))

    return loss, acc

    
