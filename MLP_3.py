import numpy as np
import helper_functions as hf
import random

class network:

    def __init__(self):
        self.layers = []
        self.if_softmax = True

    def add_layer(self, layer):
        self.layers.append(layer)

    def train_with_batch_by_epochs_num(self, trainset, val_set,  batch_size=250, learning_rate = 0.1, max_epochs=10):
        epochs_accuracy = []
        epochs_accuracy_val = []
        epochs_loss_val = []
        batches_accuracy = []

        for e in range(max_epochs):
            print('Epoka ', e)
            num_in_batch = 0
            num_of_accurate = 0
            num_of_accurate_in_batch = 0
            random.shuffle(trainset)

            for x, y in trainset:
                num_in_batch += 1
                network_output = self.forward(x)

                if (self.if_softmax):
                    network_output = hf.softmax(network_output)

                if y == self.pred(network_output):
                    num_of_accurate += 1
                    num_of_accurate_in_batch += 1
                self.backward(network_output, y, x)

                if num_in_batch == batch_size:
                    self.apply_changes(batch_size, learning_rate)
                    batches_accuracy.append(num_of_accurate_in_batch/batch_size)
                    num_in_batch = 0
                    num_of_accurate_in_batch = 0
            print('Accuracy: ', num_of_accurate / len(trainset))
            val_acc, loss = self.test_network(val_set)
            loss = loss * loss
            epochs_accuracy_val.append(val_acc)
            epochs_loss_val.append(loss)
            epochs_accuracy.append(num_of_accurate/len(trainset))

        return epochs_accuracy, epochs_accuracy_val, batches_accuracy

    def train_with_batch_by_loss_diff(self, trainset, val_set,  batch_size=250, learning_rate = 0.1, min_loss_diff = 0.001):
        epochs_accuracy = []
        epochs_accuracy_val = []
        epochs_loss_val = []
        batches_accuracy = []
        e = 0
        loss_diff = 100000
        prev_loss = 100000
        loss = 1000

        while(loss_diff > min_loss_diff):
            random.shuffle(trainset)
            e += 1
            print('Epoka ', e)
            num_in_batch = 0
            num_of_accurate = 0
            num_of_accurate_in_batch = 0

            for x, y in trainset:
                num_in_batch += 1
                network_output = self.forward(x)

                if (self.if_softmax):
                    network_output = hf.softmax(network_output)

                if y == self.pred(network_output):
                    num_of_accurate += 1
                    num_of_accurate_in_batch += 1
                self.backward(network_output, y, x)

                if num_in_batch == batch_size:
                    self.apply_changes(batch_size, learning_rate)
                    batches_accuracy.append(num_of_accurate_in_batch/batch_size)
                    num_in_batch = 0
                    num_of_accurate_in_batch = 0
            print('Accuracy: ', num_of_accurate / len(trainset))

            val_acc, loss = self.test_network(val_set)
            loss = loss * loss
            loss_diff = loss - prev_loss
            prev_loss = loss
            epochs_accuracy_val.append(val_acc)
            epochs_loss_val.append(loss)
            epochs_accuracy.append(num_of_accurate/len(trainset))

        return epochs_accuracy, epochs_accuracy_val, batches_accuracy, e

    def train_with_batch_by_loss_diff_for_experiments(self, trainset, val_set,  batch_size=250, learning_rate = 0.02, momentum_rate=0, min_loss_diff = 0.1, max_epochs = 10):
        epochs_accuracy = []
        epochs_accuracy_val = []

        e = 0
        loss_diff = 100000
        prev_loss = 0
        loss = 1000

        while loss_diff > min_loss_diff and e < max_epochs:
            random.shuffle(trainset)
            e += 1
            print(e, end=" ")
            num_in_batch = 0
            num_of_accurate = 0

            for x, y in trainset:
                num_in_batch += 1
                network_output = self.forward(x)

                if (self.if_softmax):
                    network_output = hf.softmax(network_output)

                if y == self.pred(network_output):
                    num_of_accurate += 1
                self.backward(network_output, y, x)

                if num_in_batch == batch_size:
                    self.apply_changes(batch_size, learning_rate, momentum_rate)
                    num_in_batch = 0

            val_acc, loss = self.test_network(val_set)
            loss = loss * loss
            loss_diff = loss - prev_loss
            prev_loss = loss
            epochs_accuracy_val.append(val_acc)
            epochs_accuracy.append(num_of_accurate/len(trainset))

        return e, epochs_accuracy[-1], epochs_accuracy_val[-1], epochs_accuracy, epochs_accuracy_val

    def train_with_adam_by_epochs_num(self, trainset, val_set,  batch_size=250, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, max_epochs = 7):
        epochs_accuracy = []
        epochs_accuracy_val = []

        e = 0

        while e < max_epochs:
            random.shuffle(trainset)
            e += 1
            print(e, end=" ")
            num_in_batch = 0
            num_of_accurate = 0

            for x, y in trainset:
                num_in_batch += 1
                network_output = self.forward(x)

                if (self.if_softmax):
                    network_output = hf.softmax(network_output)

                if y == self.pred(network_output):
                    num_of_accurate += 1
                self.backward(network_output, y, x)

                if num_in_batch == batch_size:
                    self.apply_changes_with_adam(batch_size, learning_rate)
                    num_in_batch = 0

            val_acc, loss = self.test_network(val_set)
            print('osiągnięta skuteczność walidacyjna', val_acc)
            epochs_accuracy_val.append(val_acc)
            epochs_accuracy.append(num_of_accurate/len(trainset))

        return e, epochs_accuracy[-1], epochs_accuracy_val[-1], epochs_accuracy, epochs_accuracy_val

    def train_with_adagrad_by_epochs_num(self, trainset, val_set,  batch_size=250, learning_rate = 0.1, max_epochs = 25):
        epochs_accuracy = []
        epochs_accuracy_val = []

        e = 0

        while e < max_epochs:
            random.shuffle(trainset)
            e += 1
            print(e, end=" ")
            num_in_batch = 0
            num_of_accurate = 0

            for x, y in trainset:
                num_in_batch += 1
                network_output = self.forward(x)

                if (self.if_softmax):
                    network_output = hf.softmax(network_output)

                if y == self.pred(network_output):
                    num_of_accurate += 1
                self.backward(network_output, y, x)

                if num_in_batch == batch_size:
                    self.apply_changes_with_adagrad(batch_size, learning_rate)
                    num_in_batch = 0

            val_acc, loss = self.test_network(val_set)
            print('osiągnięta skuteczność walidacyjna', val_acc)
            epochs_accuracy_val.append(val_acc)
            epochs_accuracy.append(num_of_accurate/len(trainset))

        return e, epochs_accuracy[-1], epochs_accuracy_val[-1], epochs_accuracy, epochs_accuracy_val


    def train_with_nesterov_by_epochs_num(self, trainset, val_set,  batch_size=250, learning_rate = 0.1, momentum_rate = 0.1, max_epochs = 6):
        epochs_accuracy = []
        epochs_accuracy_val = []

        e = 0

        while e < max_epochs:
            random.shuffle(trainset)
            e += 1
            print(e, end=" ")
            num_in_batch = 0
            num_of_accurate = 0

            for x, y in trainset:
                num_in_batch += 1
                network_output = self.forward(x)

                if (self.if_softmax):
                    network_output = hf.softmax(network_output)

                if y == self.pred(network_output):
                    num_of_accurate += 1
                self.backward_with_nesterov(network_output, y, x, momentum_rate)

                if num_in_batch == batch_size:
                    self.apply_changes(batch_size, learning_rate, momentum_rate)
                    num_in_batch = 0

            val_acc, loss = self.test_network(val_set)
            print('osiągnięta skuteczność walidacyjna', val_acc)
            epochs_accuracy_val.append(val_acc)
            epochs_accuracy.append(num_of_accurate/len(trainset))

        return e, epochs_accuracy[-1], epochs_accuracy_val[-1], epochs_accuracy, epochs_accuracy_val

    def train_with_momentum_by_epochs_num(self, trainset, val_set,  batch_size=250, learning_rate = 0.1, momentum_rate = 0.1, max_epochs = 6):
        epochs_accuracy = []
        epochs_accuracy_val = []

        e = 0

        while e < max_epochs:
            random.shuffle(trainset)
            e += 1
            print(e, end=" ")
            num_in_batch = 0
            num_of_accurate = 0

            for x, y in trainset:
                num_in_batch += 1
                network_output = self.forward(x)

                if (self.if_softmax):
                    network_output = hf.softmax(network_output)

                if y == self.pred(network_output):
                    num_of_accurate += 1
                self.backward(network_output, y, x)

                if num_in_batch == batch_size:
                    self.apply_changes(batch_size, learning_rate, momentum_rate)
                    num_in_batch = 0

            val_acc, loss = self.test_network(val_set)
            print('osiągnięta skuteczność walidacyjna', val_acc)
            epochs_accuracy_val.append(val_acc)
            epochs_accuracy.append(num_of_accurate/len(trainset))

        return e, epochs_accuracy[-1], epochs_accuracy_val[-1], epochs_accuracy, epochs_accuracy_val

    def test_network(self, test_data):
        num_of_correct = 0
        loss = 0
        for x, y in test_data:
            output = self.forward(x)
            pred = self.pred(output)
            loss += self.loss_function(output, pred)
            if(pred == y):
                num_of_correct += 1
        loss /= len(test_data)
        return num_of_correct/len(test_data), loss


    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, output, y, x):
        y_arr = self.y_to_1n(y)
        error = output - y_arr

        for index in range(len(self.layers)-1, 0, -1):
            error = self.layers[index].backward(error, self.layers[index-1].a_mem)

        self.layers[0].backward(error, x)

    def backward_with_nesterov(self, output, y, x, momentum_rate):
        y_arr = self.y_to_1n(y)
        error = output - y_arr

        for index in range(len(self.layers)-1, 0, -1):
            error = self.layers[index].backward_with_nesterov(error, self.layers[index-1].a_mem, momentum_rate)

        self.layers[0].backward(error, x)

    def apply_changes(self, batch_size, learning_rate, momentum_rate = 0):
        for l in self.layers:
            l.apply_weights_change(learning_rate, batch_size, momentum_rate)

    def apply_changes_with_adam(self, batch_size, learning_rate):
        for l in self.layers:
            l.apply_changes_adam(learning_rate, batch_size)

    def apply_changes_with_adagrad(self, batch_size, learning_rate):
        for l in self.layers:
            l.apply_changes_adagrad(learning_rate, batch_size)

    def y_to_1n(self, y):
        y_array = np.zeros_like(self.layers[-1].biases)
        y_array[y] = 1
        return y_array

    def pred(self, output):
        return np.argmax(output)

    def loss_function(self, output, y):
        y_ar = self.y_to_1n(y)
        loss = -np.sum(y_ar * np.vectorize(self.log_with_zero)(output))
        return loss

    def log_with_zero(self, x):
        if (x>0):
            return np.log(x)
        else:
            return 0
