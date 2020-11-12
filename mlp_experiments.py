import keras
import MLP_3 as mlp
import MLP as act_func
import dataMethods as data
import layer


def test_learning_rates(learning_rates, neurons_in_layers = [784, 200, 100, 10], activation = act_func.tanh, data_split = 0.9, batch_size = 150, max_epochs=5):
    train_set, valid_set = get_start_data(data_split)
    train_accuracy = []
    test_accuracy = []

    for lr in learning_rates:
        network = create_network(neurons_in_layers)
        epochs_accuracy, epochs_accuracy_val,  = network.train_with_batch(train_set, batch_size, lr, max_epochs)

    return 0


def get_start_data(data_split):
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
    train_set, valid_set = data.get_split_data(data_split, x_train, y_train)
    return train_set, valid_set

def create_network(neur_in_layers, activation_function):
    network = mlp.network()
    for i in range(len(neur_in_layers)-2):
        l = layer.Layer(neur_in_layers[i], neur_in_layers[i+1], activation_function)
        network.add_layer(l)
    l = layer.Layer(neur_in_layers[-2], neur_in_layers[-1], act_func.linear)
    network.add_layer(l)
    return network

create_network([784, 200, 100, 10], act_func.tanh)
