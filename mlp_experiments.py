import MLP_3 as mlp
import helper_functions as act_func
import data_methods as data
import layer
from datetime import datetime
import matplotlib.pyplot as plt
import experiment_helpers as eh
import time


def test_learning_rates_avg(learning_rates, neurons_in_layers = [784, 250, 10],  data_split = 0.3, valid_split = 0.1, activation = act_func.tanh, batch_size = 150, repetitions = 2, max_epochs=5):
    train_set, valid_set, test_set = eh.get_start_data(data_split, valid_split)
    train_accuracy_array = []
    valid_accuracy_array = []
    test_accuracy_array = []
    avg_time = []
    epochs_array = []

    params_text = eh.make_params_list(neurons_in_layers, activation, '', batch_size, len(train_set), len(valid_set), len(test_set))
    title = 'Badanie wplywu wspolczynnika uczenia na trenowanie sieci neuronowej'

    for lr in learning_rates:
        print('Wspolczynnik uczenia: ', lr)
        start_time = time.time()
        epochs_sum = 0
        train_accuracy_sum = 0
        valid_accuracy_sum = 0
        test_accuracy_sum = 0

        for _ in range(repetitions):
            network = eh.create_network(neurons_in_layers, activation)
            epochs, last_epoch_accuracy, last_epoch_val_accuracy, _, _ = network.train_with_batch_by_loss_diff_for_experiments(train_set.copy(), valid_set.copy(), batch_size, lr)
            test_accuracy, _ = network.test_network(test_set.copy())

            epochs_sum += epochs
            train_accuracy_sum += last_epoch_accuracy
            valid_accuracy_sum += last_epoch_val_accuracy
            test_accuracy_sum += test_accuracy

        end_time = time.time() - start_time
        end_time /= epochs_sum

        epochs_sum /= repetitions
        train_accuracy_sum /= repetitions
        valid_accuracy_sum /= repetitions
        test_accuracy_sum /= repetitions

        epochs_array.append(epochs_sum)
        train_accuracy_array.append(train_accuracy_sum)
        valid_accuracy_array.append(valid_accuracy_sum)
        test_accuracy_array.append(test_accuracy_sum)
        avg_time.append(end_time)

    return epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, avg_time, params_text, title


def run_experiment_with_learning_rates(learning_rates = [0.02, 0.07, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7]):
    epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, end_time_array, params_text, title = test_learning_rates_avg(learning_rates)
    text = eh.matricesForTexTable([learning_rates, epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, end_time_array])
    eh.save_text_to_file(r'results\learning_rate\learning_rate_experiments.txt', text, title, params_text)
    plt.plot(learning_rates, train_accuracy_array, 'go', learning_rates, valid_accuracy_array, 'yo', learning_rates, test_accuracy_array, 'ro')
    plt.legend(['Train accuracy', 'Validation accuracy', 'Test accuracy'])
    plt.xlabel('Learning rate')
    plt.title('Accuracy(learning rate)')
    datetimeStr = datetime.now().strftime("%d-%m-%Y-%H%M%S")
    plt.savefig(r'results\learning_rate\L'+datetimeStr+'-acc.png')
    plt.close()

    plt.plot(learning_rates, epochs_array, 'ro')
    plt.xlabel('Learning rate')
    plt.ylabel('Epochs')
    plt.title('Epochs(learning rate)')
    plt.savefig(r'results\learning_rate\L' + datetimeStr + '-ep.png')
    plt.close()

    plt.plot(learning_rates, end_time_array, 'ro')
    plt.xlabel('Learning rate')
    plt.ylabel('Time')
    plt.title('Time(learning rate)')
    plt.savefig(r'results\learning_rate\L' + datetimeStr + '-time.png')
    plt.close()

def test_batch_sizes_avg(batch_sizes, neurons_in_layers = [784, 250, 10],  data_split = 0.3, valid_split = 0.1, activation = act_func.tanh, learning_rate = 0.2, repetitions = 10, max_epochs=5):
    train_set, valid_set, test_set = eh.get_start_data(data_split, valid_split)
    train_accuracy_array = []
    valid_accuracy_array = []
    test_accuracy_array = []
    epochs_array = []
    avg_time = []

    params_text = eh.make_params_list(neurons_in_layers, activation, learning_rate, '', len(train_set), len(valid_set), len(test_set))
    title = 'Badanie wpływu rozmiaru paczki na trenowanie sieci neuronowej'

    for bs in batch_sizes:
        start_time = time.time()
        print('Rozmiar paczki: ', bs)
        epochs_sum = 0
        train_accuracy_sum = 0
        valid_accuracy_sum = 0
        test_accuracy_sum = 0

        for _ in range(repetitions):
            network = eh.create_network(neurons_in_layers, activation)
            epochs, last_epoch_accuracy, last_epoch_val_accuracy, _, _ = network.train_with_batch_by_loss_diff_for_experiments(train_set.copy(), valid_set.copy(), bs, learning_rate)
            test_accuracy, _ = network.test_network(test_set.copy())

            epochs_sum += epochs
            train_accuracy_sum += last_epoch_accuracy
            valid_accuracy_sum += last_epoch_val_accuracy
            test_accuracy_sum += test_accuracy

        end_time = time.time() - start_time
        end_time /= epochs_sum

        epochs_sum /= repetitions
        train_accuracy_sum /= repetitions
        valid_accuracy_sum /= repetitions
        test_accuracy_sum /= repetitions

        epochs_array.append(epochs_sum)
        train_accuracy_array.append(train_accuracy_sum)
        valid_accuracy_array.append(valid_accuracy_sum)
        test_accuracy_array.append(test_accuracy_sum)
        avg_time.append(end_time)

    return epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, avg_time, params_text, title


def run_experiment_with_batch_sizes(batch_sizes = [20, 50, 100, 150, 250, 500]):
    epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, end_time_array, params_text, title = test_batch_sizes_avg(batch_sizes)
    text = eh.matricesForTexTable([batch_sizes, epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, end_time_array])
    eh.save_text_to_file(r'results\batch_size\batch_size_experiments.txt', text, title, params_text)
    plt.plot(batch_sizes, train_accuracy_array, 'go', batch_sizes, valid_accuracy_array, 'yo', batch_sizes,
             test_accuracy_array, 'ro')
    plt.legend(['Train accuracy', 'Validation accuracy', 'Test accuracy'])
    plt.xlabel('Batch size')
    plt.title('Accuracy(batch size)')
    datetimeStr = datetime.now().strftime("%d-%m-%Y-%H%M%S")
    plt.savefig(r'results\batch_size\R' + datetimeStr + '-acc.png')
    plt.close()

    plt.plot(batch_sizes, epochs_array, 'ro')
    plt.xlabel('Batch size')
    plt.ylabel('Epochs')
    plt.title('Epochs(batch size)')
    plt.savefig(r'results\batch_size\L' + datetimeStr + '-ep.png')
    plt.close()

    plt.plot(batch_sizes, end_time_array, 'ro')
    plt.xlabel('Batch size')
    plt.ylabel('Time')
    plt.title('Time(batch size)')
    plt.savefig(r'results\batch_size\L' + datetimeStr + '-time.png')
    plt.close()


def test_activation_functions_avg(activation_functions = [act_func.tanh, act_func.sigmoid, act_func.relu], neurons_in_layers = [784, 200, 100, 10],  data_split = 0.2, valid_split = 0.1, batch_size=200, learning_rate = 0.2, repetitions = 2, max_epochs=5):
    train_set, valid_set, test_set = eh.get_start_data(data_split, valid_split)
    train_accuracy_array = []
    valid_accuracy_array = []
    test_accuracy_array = []
    epochs_array = []
    avg_time = []

    params_text = eh.make_params_list(neurons_in_layers, '', learning_rate, '', len(train_set), len(valid_set), len(test_set))
    title = 'Badanie wpływu funkcji aktywacji na trenowanie sieci neuronowej'

    for af in activation_functions:
        start_time = time.time()
        print('Funkcja aktywacji: ', act_func.activationString(af))
        epochs_sum = 0
        train_accuracy_sum = 0
        valid_accuracy_sum = 0
        test_accuracy_sum = 0

        for _ in range(repetitions):
            network = eh.create_network(neurons_in_layers, af)
            epochs, last_epoch_accuracy, last_epoch_val_accuracy, _, _ = network.train_with_batch_by_loss_diff_for_experiments(train_set.copy(), valid_set.copy(), batch_size, learning_rate)
            test_accuracy, _ = network.test_network(test_set.copy())

            epochs_sum += epochs
            train_accuracy_sum += last_epoch_accuracy
            valid_accuracy_sum += last_epoch_val_accuracy
            test_accuracy_sum += test_accuracy

        end_time = time.time() - start_time
        end_time /= epochs_sum

        epochs_sum /= repetitions
        train_accuracy_sum /= repetitions
        valid_accuracy_sum /= repetitions
        test_accuracy_sum /= repetitions

        epochs_array.append(epochs_sum)
        train_accuracy_array.append(train_accuracy_sum)
        valid_accuracy_array.append(valid_accuracy_sum)
        test_accuracy_array.append(test_accuracy_sum)
        avg_time.append(end_time)

    return epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, avg_time, params_text, title


def run_experiment_activation_functions(batch_sizes = [20, 50, 100, 150, 250, 500]):
    epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, end_time_array, params_text, title = test_batch_sizes_avg(batch_sizes)
    text = eh.matricesForTexTable([batch_sizes, epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, end_time_array])
    eh.save_text_to_file(r'results\batch_size\batch_size_experiments.txt', text, title, params_text)
    plt.plot(batch_sizes, train_accuracy_array, 'go', batch_sizes, valid_accuracy_array, 'yo', batch_sizes,
             test_accuracy_array, 'ro')
    plt.legend(['Train accuracy', 'Validation accuracy', 'Test accuracy'])
    plt.xlabel('Batch size')
    plt.title('Accuracy(batch size)')
    datetimeStr = datetime.now().strftime("%d-%m-%Y-%H%M%S")
    plt.savefig(r'results\batch_size\R' + datetimeStr + '-acc.png')
    plt.close()

    plt.plot(batch_sizes, epochs_array, 'ro')
    plt.xlabel('Batch size')
    plt.ylabel('Epochs')
    plt.title('Epochs(batch size)')
    plt.savefig(r'results\batch_size\L' + datetimeStr + '-ep.png')
    plt.close()

    plt.plot(batch_sizes, end_time_array, 'ro')
    plt.xlabel('Batch size')
    plt.ylabel('Time')
    plt.title('Time(batch size)')
    plt.savefig(r'results\batch_size\L' + datetimeStr + '-time.png')
    plt.close()

def test_neurons_nums(neuron_nums = [100, 10, 20, 10],  data_split = 0.2, valid_split = 0.1, batch_size=150, activation = act_func.tanh, learning_rate = 0.2, momentum_rate = 0.1, repetitions = 2, max_epochs=5):
    train_set, valid_set, test_set = eh.get_start_data(data_split, valid_split)
    train_accuracy_array = []
    valid_accuracy_array = []
    test_accuracy_array = []
    epochs_array = []
    avg_time = []

    params_text = eh.make_params_list('', activation, learning_rate, batch_size, len(train_set), len(valid_set), len(test_set))
    title = 'Badanie wpływu liczby neuronów w warstwie ukrytej na trenowanie sieci neuronowej'

    for nn in neuron_nums:
        start_time = time.time()
        network_shape = [784, nn, 10]
        print('Liczba neuronów: ', nn)
        epochs_sum = 0
        train_accuracy_sum = 0
        valid_accuracy_sum = 0
        test_accuracy_sum = 0

        for _ in range(repetitions):
            network = eh.create_network(network_shape, activation)
            epochs, last_epoch_accuracy, last_epoch_val_accuracy, _, _ = network.train_with_batch_by_loss_diff_for_experiments(train_set.copy(), valid_set.copy(), batch_size, learning_rate, momentum_rate)
            test_accuracy, _ = network.test_network(test_set.copy())

            epochs_sum += epochs
            train_accuracy_sum += last_epoch_accuracy
            valid_accuracy_sum += last_epoch_val_accuracy
            test_accuracy_sum += test_accuracy

        end_time = time.time() - start_time
        end_time /= epochs_sum

        epochs_sum /= repetitions
        train_accuracy_sum /= repetitions
        valid_accuracy_sum /= repetitions
        test_accuracy_sum /= repetitions

        epochs_array.append(epochs_sum)
        train_accuracy_array.append(train_accuracy_sum)
        valid_accuracy_array.append(valid_accuracy_sum)
        test_accuracy_array.append(test_accuracy_sum)
        avg_time.append(end_time)

    return epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, avg_time, params_text, title


def run_experiment_with_neurons_nums(neurons_num = [700, 500, 200, 100, 40]):
    epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, end_time_array, params_text, title = test_neurons_nums(neurons_num)
    text = eh.matricesForTexTable([neurons_num, epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, end_time_array])
    eh.save_text_to_file(r'results\neurons_num\neurons_num_experiments.txt', text, title, params_text)
    plt.plot(neurons_num, train_accuracy_array, 'go', neurons_num, valid_accuracy_array, 'yo', neurons_num,
             test_accuracy_array, 'ro')
    plt.legend(['Train accuracy', 'Validation accuracy', 'Test accuracy'])
    plt.xlabel('Neurons num')
    plt.title('Accuracy(neurons num)')
    datetimeStr = datetime.now().strftime("%d-%m-%Y-%H%M%S")
    plt.savefig(r'results\neurons_num\R' + datetimeStr + '-acc.png')
    plt.close()

    plt.plot(neurons_num, epochs_array, 'ro')
    plt.xlabel('Neurons num')
    plt.ylabel('Epochs')
    plt.title('Epochs(neurons num)')
    plt.savefig(r'results\neurons_num\R' + datetimeStr + '-ep.png')
    plt.close()

    plt.plot(neurons_num, end_time_array, 'ro')
    plt.xlabel('Neurons num')
    plt.ylabel('Time')
    plt.title('Time(neurons num)')
    plt.savefig(r'results\neurons_num\R' + datetimeStr + '-time.png')
    plt.close()

def test_split_avg(data_splits = 0.2, neurons_in_layers = [784, 200, 10], batch_size = 150,  valid_split = -1, activation = act_func.tanh, learning_rate = 0.2, repetitions = 10, max_epochs=5):
    train_accuracy_array = []
    valid_accuracy_array = []
    test_accuracy_array = []
    epochs_array = []
    avg_time = []

    params_text = eh.make_params_list(neurons_in_layers, activation, learning_rate, batch_size, '', '', '')
    title = 'Badanie wpływu wielkości zbioru uczacego na trenowanie sieci neuronowej'

    for ds in data_splits:
        train_set, valid_set, test_set = eh.get_start_data(ds, -1)
        start_time = time.time()
        print('Data split: ', ds)
        epochs_sum = 0
        train_accuracy_sum = 0
        valid_accuracy_sum = 0
        test_accuracy_sum = 0

        for _ in range(repetitions):
            network = eh.create_network(neurons_in_layers, activation)
            epochs, last_epoch_accuracy, last_epoch_val_accuracy, _, _ = network.train_with_batch_by_loss_diff_for_experiments(train_set.copy(), valid_set.copy(), batch_size, learning_rate)
            test_accuracy, _ = network.test_network(test_set.copy())

            epochs_sum += epochs
            train_accuracy_sum += last_epoch_accuracy
            valid_accuracy_sum += last_epoch_val_accuracy
            test_accuracy_sum += test_accuracy

        end_time = time.time() - start_time
        end_time /= epochs_sum

        epochs_sum /= repetitions
        train_accuracy_sum /= repetitions
        valid_accuracy_sum /= repetitions
        test_accuracy_sum /= repetitions

        epochs_array.append(epochs_sum)
        train_accuracy_array.append(train_accuracy_sum)
        valid_accuracy_array.append(valid_accuracy_sum)
        test_accuracy_array.append(test_accuracy_sum)
        avg_time.append(end_time)

    return epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, avg_time, params_text, title


def run_experiment_with_data_split(data_splits = [0.05, 0.1, 0.2, 0.4, 0.5, 0.7, 1]):
    epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, end_time_array, params_text, title = test_split_avg(data_splits)
    text = eh.matricesForTexTable([data_splits, epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, end_time_array])
    eh.save_text_to_file(r'results\train_set_size\train_set_size_experiments.txt', text, title, params_text)
    plt.plot(data_splits, train_accuracy_array, 'go', data_splits, valid_accuracy_array, 'yo', data_splits,
             test_accuracy_array, 'ro')
    plt.legend(['Train accuracy', 'Validation accuracy', 'Test accuracy'])
    plt.xlabel('Train set split')
    plt.title('Accuracy(Train set split)')
    datetimeStr = datetime.now().strftime("%d-%m-%Y-%H%M%S")
    plt.savefig(r'results\train_set_size\T' + datetimeStr + '-acc.png')
    plt.close()

    plt.plot(data_splits, epochs_array, 'ro')
    plt.xlabel('Train set split')
    plt.ylabel('Epochs')
    plt.title('Epochs(Train set split)')
    plt.savefig(r'results\train_set_size\T' + datetimeStr + '-ep.png')
    plt.close()

    plt.plot(data_splits, end_time_array, 'ro')
    plt.xlabel('Train set split')
    plt.ylabel('Time')
    plt.title('Time(Train set split)')
    plt.savefig(r'results\train_set_size\T' + datetimeStr + '-time.png')
    plt.close()


def test_weights_ranges(weights_abs = [(0, 0.1), (0.2), (0, 0.01), (-0.1, 0), (-0.2, 0.2)], activation = act_func.tanh, network_shape=[784, 100, 10], data_split = 0.3, valid_split = 0.1, batch_size=50,  learning_rate = 0.2, repetitions = 10, max_epochs=5):
    train_set, valid_set, test_set = eh.get_start_data(data_split, valid_split)
    train_accuracy_array = []
    valid_accuracy_array = []
    test_accuracy_array = []
    epochs_array = []
    avg_time = []

    params_text = eh.make_params_list(network_shape, activation, learning_rate, batch_size, len(train_set), len(valid_set), len(test_set))
    title = 'Badanie wpływu zakresu wag początkowych na trenowanie sieci neuronowej'

    for min_w, max_w in weights_abs:
        start_time = time.time()
        print('Przedział: ', min_w)
        epochs_sum = 0
        train_accuracy_sum = 0
        valid_accuracy_sum = 0
        test_accuracy_sum = 0

        for _ in range(repetitions):
            network = eh.create_network(network_shape, activation, min_w, max_w)
            epochs, last_epoch_accuracy, last_epoch_val_accuracy, _, _ = network.train_with_batch_by_loss_diff_for_experiments(train_set.copy(), valid_set.copy(), batch_size, learning_rate)
            test_accuracy, _ = network.test_network(test_set.copy())

            epochs_sum += epochs
            train_accuracy_sum += last_epoch_accuracy
            valid_accuracy_sum += last_epoch_val_accuracy
            test_accuracy_sum += test_accuracy

        end_time = time.time() - start_time
        end_time /= epochs_sum

        epochs_sum /= repetitions
        train_accuracy_sum /= repetitions
        valid_accuracy_sum /= repetitions
        test_accuracy_sum /= repetitions

        epochs_array.append(epochs_sum)
        train_accuracy_array.append(train_accuracy_sum)
        valid_accuracy_array.append(valid_accuracy_sum)
        test_accuracy_array.append(test_accuracy_sum)
        avg_time.append(end_time)

    return epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, avg_time, params_text, title


def run_experiment_with_weights(weights_abs = [(0, 0.01), (0, 0.1), (0, 0.2),  (-0.1, 0), (-0.2, 0.2)], fun = act_func.tanh):
    epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, end_time_array, params_text, title = test_weights_ranges(weights_abs, fun)
    text = eh.matricesForTexTable([weights_abs, epochs_array, train_accuracy_array, valid_accuracy_array, test_accuracy_array, end_time_array])
    eh.save_text_to_file(r'results\initial_weights\initial_weights_experiments.txt', text, title, params_text)

    diff = eh.initial_weights_to_diff(weights_abs)

    weights_abs_to_string = eh.weights_list_to_string(weights_abs)

    plt.plot(diff, train_accuracy_array, 'go', diff, valid_accuracy_array, 'yo', diff,
             test_accuracy_array, 'ro')
    plt.legend(['Train accuracy', 'Validation accuracy', 'Test accuracy'])
    plt.xlabel('|Initial weight range|')
    plt.title('Accuracy(|Initial weight range|)')
    datetimeStr = datetime.now().strftime("%d-%m-%Y-%H%M%S")
    plt.savefig(r'results\initial_weights\R' + datetimeStr + '-acc.png')
    plt.close()

    plt.plot(diff, epochs_array, 'ro')
    plt.xlabel('|Initial weight range|')
    plt.ylabel('Epochs')
    plt.title('Epochs(|Initial weight range|)')
    plt.savefig(r'results\initial_weights\R' + datetimeStr + '-ep.png')
    plt.close()

    plt.plot(diff, end_time_array, 'ro')
    plt.xlabel('|Initial weight range|')
    plt.ylabel('Time')
    plt.title('Time(|Initial weight range|)')
    plt.savefig(r'results\initial_weights\R' + datetimeStr + '-time.png')
    plt.close()

#run_experiment_with_batch_sizes([5, 10, 50, 100, 200, 300, 500, 1000])
#run_experiment_with_learning_rates([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)
run_experiment_with_neurons_nums([20, 50])
#run_experiment_with_data_split([0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.95])

# run_experiment_with_learning_rates([0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
# run_experiment_with_batch_sizes([5, 10, 50, 100, 200, 300, 500, 1000])
# run_experiment_with_weights([(0, 0.05),(0, 0.1), (0, 0.2), (0, 0.3), (0, 0.5), (0, 0.7), (0, 1)])
#run_experiment_with_weights([(-0.05, 0.05), (-0.1, 0.1), (-0.2, 0.2), (-0.3, 0.3)])

# run_experiment_with_weights([(0, 0.05),(0, 0.1), (0, 0.2), (0, 0.3), (0, 0.5), (0, 0.7), (0, 1)], act_func.relu)
# run_experiment_with_weights([(-0.05, 0.05), (-0.1, 0.1), (-0.2, 0.2), (-0.3, 0.3)], act_func.relu)

