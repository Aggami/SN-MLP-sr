import numpy as np
import math
import random

def last_layer_error(y, pred):
        return (pred - y_to_1n(y))

def y_to_1n(y):
    y_1n = np.zeros(shape=(10, 1))
    y_1n[y][0] = 1
    return y_1n

def negative_log_likelihood(y, pred):
    return np.sum(-1*np.log(pred)*y)

def initialiseWeights(input_size, output_size, min_w=0, max_w=0.1):
    weights = (max_w - min_w) * np.random.normal(size = (input_size, output_size)) + min_w
    #weights = np.random.uniform(size = (input_size, output_size))
    return weights

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def relu(x):
    return np.maximum(0, x)


def activationDerivative(actFun):
    if (actFun == sigmoid):
        return sigmoidDerivative
    if (actFun == tanh):
        return tanhDerivative
    if (actFun == relu):
        return np.vectorize(reluDerivative)
    if(actFun == linear):
        return linearDerivative

def activationString(actFun):
    if (actFun == sigmoid):
        return 'sigmoid'
    if (actFun == tanh):
        return 'tanh'
    if (actFun == relu):
        return 'relu'
    if(actFun == linear):
        return 'linear'


def sigmoidDerivative(y):
    y_s = sigmoid(y)
    return y_s * (1 - y_s)

def tanhDerivative(y):
    y2 = tanh(y)
    return 1-(y2*y2)

# def reluDerivative(y):
#     return (y > 0) * 1

def reluDerivative(x: float) -> float:
    return 1 if x > 0 else 0

def clear_array_list(x):
    for i in range(len(x)):
        x[i] = x[i] * 0
    return x

def linear(x):
    return x

def linearDerivative(x):
    return 1
