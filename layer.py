import numpy as np
import MLP as mlp

class Layer:
    a_mem = np.array([])
    z_mem = np.array([])
    weights = np.array([])
    biases = np.array([])
    activation_function = mlp.sigmoid
    derivative_function = mlp.activationDerivative(activation_function)

    def __init__(self, input_num, output_num, activation_fun):
        self.weights = mlp.initialiseWeights(input_num, output_num, 0, 0.1)
        self.biases = mlp.initialiseWeights(output_num, 1, 0, 0.1)
        self.activation_function = activation_fun
        self.derivative_function = mlp.activationDerivative(activation_fun)
        self.total_weights_change = np.zeros_like(self.weights)
        self.total_biases_change = np.zeros_like(self.biases)

    def forward(self, x):
        self.z_mem = np.dot(self.weights.T, x) + self.biases
        self.a_mem = self.activation_function(self.z_mem)
        return self.a_mem

    def forward_2(self, x):
        self.z_mem = np.dot(self.weights.T, x) + self.biases
        self.a_mem = self.activation_function(self.z_mem)
        return self.a_mem

    def backward(self, error, a_prev):
        error = np.multiply(error, np.vectorize(self.derivative_function)(self.z_mem))
        next_error = np.dot(self.weights, error)
        helper = np.dot(error, a_prev.T)
        self.total_weights_change += helper.T
        self.total_biases_change += error

        return next_error

    def apply_weights_change(self, learning_rate, batch_size):
        self.weights -= learning_rate / batch_size * self.total_weights_change
        self.biases -= learning_rate / batch_size * self.total_biases_change



