import numpy as np
import math
import random

class MLP:

    def __init__(self):
        self.layers = []
        self.prev_error = 100000
        self.curr_error = 0
        self.epochs = 0
        self.if_softmax = False
        self.total_weights_change = np.array([])
        self.total_biases_change = np.array([])
        self.learning_rate = 0.2

    def addLayer(self, layer):
        self.layers.append(layer)

    def initialise_total_change(self):
        self.total_weights_change = []
        self.total_biases_change = []
        for l in self.layers:
            self.total_weights_change.append(np.zeros_like(l.weights))
            self.total_biases_change.append(np.zeros_like(l.biases))

    def predict(self, x):

        nextX = x
        i = 0
        for l in self.layers:
            print('Warstwa ', i, ': ')
            print(nextX)
            nextX = l.forward(nextX)

        print('Wynik: ', nextX)


        print('Suma po softmax', sum(nextX))
        pred = nextX.argmax()

        print('Predykcja: ',pred )
        return pred

    def forward(self, x):
        for l in self.layers:
            nextX = l.forward(nextX)




    def train(self, trainset, maxE, max_epochs=10):
        #poprawic warunek
        #while (self.prev_error - self.curr_error) * (self.prev_error - self.curr_error) > maxE | self.epochs < max_epochs:
        while self.epochs < max_epochs:
            self.prev_error = self.curr_error
            self.curr_error = 0
            self.epochs += 1
            self.train_one_epoch(trainset)


    def train_one_epoch(self, trainset):
        num_of_errors = 0
        random.shuffle(trainset)
        for x, y in trainset:
            pred = self.forwardPropagation(x)
            y_pred = pred.argmax()
            if(y != y_pred):
                num_of_errors += 1
            error = last_layer_error(y, pred)

            self.backward_propagation(error)

        print('Liczba błędow w epoce: ', num_of_errors)
        return num_of_errors

    def forwardPropagation(self, x):
        nextX = x
        i = 0
        for l in self.layers:
            nextX = l.forward(nextX)
        #nextX = softmax(nextX)
        return nextX

    def forwardPropagationWithBatch(self, x):
        nextX = x
        i = 0
        for l in self.layers:
            nextX = l.forward_with_batch(nextX)
        return nextX

    def backward_propagation(self, error):
        for i in range(len(self.layers)-1, 0, -1):
            error = self.layers[i].backward_propagation(error)

    def backward_propagation_with_batch (self, error):
        for i in range(len(self.layers)-1, 0, -1):
            error = self.layers[i].backward_propagation_with_batch(error)

    def backpropagation_with_minibatch(self, pred, y):
        error = np.subtract(pred, y)

        self.total_weights_change[-1] += np.dot(error, self.layers[-2].neurons.T).T
        self.total_biases_change[-1] += error

        for i in range (len(self.layers) - 2, 1, -1):
            der = activationDerivative(self.layers[i].activationFunction)
            neurDer = np.vectorize(der)(self.layers[i].zRem)
            next_weights = self.layers[i+1].weights
            error = np.multiply(np.dot(next_weights, error), neurDer)
            self.total_weights_change[i] += np.dot(error, self.layers[i].input.T).T
            self.total_biases_change[i] += error

        return 0

    def update_weights(self, batch_size):
        #print(self.total_weights_change)
        for layer_index in range(len(self.layers) - 1, 0, -1):
            self.layers[layer_index].weights += (self.learning_rate / batch_size) * self.total_weights_change[layer_index]
            self.layers[layer_index].biases += (self.learning_rate / batch_size) * self.total_biases_change[layer_index]


    def clean_totals(self):
        self.total_biases_change = clear_array_list(self.total_biases_change)
        self.total_weights_change = clear_array_list(self.total_weights_change)

    def train_network_with_minibatch(self, trainset, batch_size = 10, epochs = 20):
        self.initialise_total_change()

        for _ in range(epochs):
            results = []

            print('Epoka 1')
            num_of_correct = 0
            num_in_batch = 1
            for x, y in trainset:

                y_array = y_to_1n(y)
                next_x = x
                num_in_batch += 1
                for l in self.layers:
                    next_x = l.forward(next_x)

                if (self.if_softmax):
                    next_x = softmax(next_x)

                pred = next_x.argmax()
                results.append((pred, y))

                if (pred == y):
                    num_of_correct += 1


                self.backpropagation_with_minibatch(next_x, y_array)


                if num_in_batch == batch_size:
                    self.update_weights(batch_size)
                    self.clean_totals()
                    num_in_batch = 1

            #print(results)
            print('Accuracy: ', num_of_correct / len(trainset))












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
        return reluDerivative
    if(actFun == linear):
        return linearDerivative


def sigmoidDerivative(y):
    return sigmoid(y) * (1 - sigmoid(y))

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