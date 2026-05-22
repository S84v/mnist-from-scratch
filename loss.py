import numpy as np

def cross_entropy_loss(y_pred, y_true):
    n = y_pred.shape[0]

    loss = -np.log(y_pred[np.arange(n), y_true] + 1e-9)

    return np.mean(loss)