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
    mask = label_mask.reshape((label_mask.shape[0], label_mask.shape[1]))

    loss = np.multiply(y_true, np.log(y_pred))
    loss = np.multiply(loss, label_mask)
    frames = np.sum(mask, axis=-1).astype(np.int)
    loss = -np.sum(np.sum(loss, axis=-1), axis=-1)

    """ accuracy """
    correct = np.equal(np.argmax(y_true, axis=-1),
                       np.argmax(y_pred, axis=-1)).astype(np.float)
    correct = np.multiply(correct, mask)
    correct = np.sum(correct, axis=-1)
    
    losses=[]
    accs=[]
    for f in range(frames.shape[0]):
        if frames[f] > 0:
            losses.append(loss[f]/frames[f])
            accs.append(correct[f]/frames[f])

    return losses, accs

    
