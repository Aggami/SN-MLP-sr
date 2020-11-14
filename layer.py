import numpy as np
import helper_functions as hf

class Layer:
    a_mem = np.array([])
    z_mem = np.array([])

    def __init__(self, input_num, output_num, activation_fun, min_w = 0, max_w = 0.1, if_momentum = False):
        self.weights = hf.initialiseWeights(input_num, output_num, min_w, max_w)
        self.biases = hf.initialiseWeights(output_num, 1, min_w, max_w)
        self.activation_function = activation_fun
        self.derivative_function = hf.activationDerivative(activation_fun)
        self.total_weights_change = np.zeros_like(self.weights)
        self.total_biases_change = np.zeros_like(self.biases)
        self.total_weights_change_square = np.zeros_like(self.weights)
        self.total_biases_change_square = np.zeros_like(self.biases)


        self.if_momentum = if_momentum
        self.momentum_weights = np.zeros_like(self.weights)
        self.momentum_biases = np.zeros_like(self.biases)

        self.momentum_weights_square = np.zeros_like(self.weights)
        self.momentum_biases_square = np.zeros_like(self.biases)

        self.gradients_sum_w = np.zeros_like(self.weights)
        self.gradients_sum_b = np.zeros_like(self.biases)


    def forward(self, x):
        self.z_mem = np.dot(self.weights.T, x) + self.biases
        self.a_mem = self.activation_function(self.z_mem)
        return self.a_mem

    def backward(self, error, a_prev):
        error = np.multiply(error, self.derivative_function(self.z_mem))
        next_error = np.dot(self.weights, error)
        helper = np.dot(error, a_prev.T)
        self.total_weights_change += helper.T
        self.total_biases_change += error

        self.total_weights_change_square += self.total_weights_change ** 2
        self.total_biases_change_square += self.total_biases_change ** 2

        return next_error

    def backward_with_nesterov(self, error, a_prev, mom_rate):
        error = np.multiply(error, self.derivative_function(self.z_mem))
        next_error = np.dot(self.weights, error)
        helper = np.dot(error, a_prev.T)
        self.total_weights_change += helper.T - mom_rate * self.momentum_weights
        self.total_biases_change += error - mom_rate * self.momentum_biases

        return next_error

    def apply_weights_nesterov(self, learning_rate, batch_size, momentum_rate = 0):
        weights_change = learning_rate / batch_size * self.total_weights_change
        biases_change = learning_rate / batch_size * self.total_biases_change

        weights_change += momentum_rate / batch_size * self.momentum_weights
        biases_change += momentum_rate / batch_size * self.momentum_biases
        self.momentum_weights = weights_change
        self.momentum_biases = biases_change

        self.weights -= weights_change
        self.biases -= biases_change
        self.total_weights_change *= 0
        self.total_biases_change *= 0


    def apply_weights_change(self, learning_rate, batch_size, momentum_rate = 0):
        weights_change = learning_rate / batch_size * self.total_weights_change
        biases_change = learning_rate / batch_size * self.total_biases_change

        if (self.if_momentum):
            weights_change += momentum_rate / batch_size * self.momentum_weights
            biases_change += momentum_rate / batch_size * self.momentum_biases
            self.momentum_weights = weights_change
            self.momentum_biases = biases_change

        self.weights -= weights_change
        self.biases -= biases_change
        self.total_weights_change *= 0
        self.total_biases_change *= 0

    def apply_changes_adam(self, learning_rate, batch_size, beta1 = 0.9, beta2 = 0.999, e = 1e-8):
        self.momentum_weights = beta1 * self.momentum_weights + (1-beta1) * self.total_weights_change
        self.momentum_biases = beta1 * self.momentum_biases + (1-beta1) * self.total_biases_change

        self.momentum_weights_square = beta2 * self.momentum_weights_square + (1 - beta2) * self.total_weights_change * self.total_weights_change
        self.momentum_biases_square = beta2 * self.momentum_biases_square + (1 - beta2) * self.total_biases_change * self.total_biases_change

        m_norm_w = self.momentum_weights / (1 - beta1)
        m_norm_b = self.momentum_biases / (1 - beta1)

        v_norm_w = self.momentum_weights_square / (1-beta2)
        v_norm_b = self.momentum_biases_square / (1-beta2)

        weights_change = learning_rate/batch_size * m_norm_w / (np.sqrt(v_norm_w) + e)
        biases_change = learning_rate/batch_size * m_norm_b / (np.sqrt(v_norm_b) + e)

        self.weights -= weights_change
        self.biases -= biases_change
        self.total_weights_change *= 0
        self.total_biases_change *= 0


    def apply_changes_adagrad(self, learning_rate, batch_size, e = 1e-8):

        weights_change = learning_rate / batch_size / (np.sqrt(self.gradients_sum_w + e)) * self.total_weights_change
        biases_change = learning_rate / batch_size / (np.sqrt(self.gradients_sum_b + e)) * self.total_biases_change

        # self.gradients_sum_w += self.total_weights_change_square / batch_size
        # self.gradients_sum_b += self.total_biases_change_square / batch_size

        self.gradients_sum_w += self.total_weights_change * self.total_weights_change / batch_size
        self.gradients_sum_b += self.total_biases_change * self.total_biases_change / batch_size

        self.weights -= weights_change
        self.biases -= biases_change
        self.total_weights_change *= 0
        self.total_biases_change *= 0
