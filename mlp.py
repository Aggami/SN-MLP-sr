import numpy as np
import helper_functions as mlp

class network:
    layers = []
    epochs = 0
    if_softmax = True

    def add_layer(self, layer):
        self.layers.append(layer)

    def train_with_batch(self, trainset, val_set,  batch_size=250, learning_rate = 0.01, max_epochs=3):
        epochs_accuracy = []
        epochs_accuracy_val = []
        epochs_loss_val = []
        batches_accuracy = []

        for e in range(max_epochs):
            print('Epoka ', e)
            num_in_batch = 0
            num_of_accurate = 0
            num_of_accurate_in_batch = 0

            for x, y in trainset:
                num_in_batch += 1
                network_output = self.forward(x)

                if (self.if_softmax):
                    network_output = mlp.softmax(network_output)

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

    def apply_changes(self, batch_size, learning_rate):
        for l in self.layers:
            l.apply_weights_change(learning_rate, batch_size)

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
