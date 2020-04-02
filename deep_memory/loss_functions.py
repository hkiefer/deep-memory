from __future__ import division
import numpy as np
from deep_memory import *

class Loss(object):
    def loss(self, y, y_pred):
        return np.absolute(y_pred - y)

    def gradient(self, y, y_pred):
        return (y_pred - y)/np.absolute(y_pred - y) 

    def acc(self, y, y_pred):
        return 0

class SquareLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5 * np.power((y - y_pred), 2)

    def gradient(self, y, y_pred):
        return -(y - y_pred)
    
class APE(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return (y-y_pred)/y

    def gradient(self, y, y_pred):
        return -(1/y)
    
class SquareLogLoss(Loss):
    def __init__(self): pass

    def loss(self, y, y_pred):
        return 0.5*np.power((np.log(y-1) - np.log(y_pred-1)), 2)

    def gradient(self, y, y_pred):
        return -1/(y_pred-1)*(np.log(y-1) - np.log(y_pred-1))

class CrossEntropy(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))

    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)


