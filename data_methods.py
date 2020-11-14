import random
import numpy as np

# def get_mnist():
#     (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
#     return (x_train, y_train), (x_test, y_test)

def get_data_from_files():
    path = '/mnist/'
    train_data = np.loadtxt(r"C:\Users\Aga\PycharmProjects\cw2-mlp\sn_mlp_sr-Aggami\mnist\mnist_train.csv",
                            delimiter=",")
    test_data = np.loadtxt(r"C:\Users\Aga\PycharmProjects\cw2-mlp\sn_mlp_sr-Aggami/mnist/mnist_test.csv" ,
                           delimiter=",")
    train_data = divide_set_in_x_and_y(train_data)
    test_data = divide_set_in_x_and_y(test_data)

    return train_data, test_data

def get_data_from_files_with_vaildation_data(train_perc, val_perc = -1):
    train_data, test_data = get_data_from_files()
    val_start_index = int(len(train_data) * train_perc)

    train_set = train_data[0:val_start_index]
    if (val_perc == -1):
        val_set = train_data[val_start_index:]
    else:
        val_end_index = int(val_start_index + val_perc * len(train_data))
        val_set = train_data[val_start_index:val_end_index]

    return train_set, val_set, test_data


def divide_set_in_x_and_y(set):

    x_y = []
    for s in set:
        y = int(s[0])
        x = np.array(s[1:]/255).reshape((784, 1))
        x_y.append((x, y))

    return x_y


def get_train_data(num, x_train, y_train):
    trainset = []
    for i in range(num):
        to_add = (x_train[i].reshape(784, -1)/255, y_train[i])
        trainset.append(to_add)
    return trainset


def get_split_data_with_test(train_perc, valid_perc, x_train, y_train):
    length = len(x_train)
    train_num = int(length * train_perc)
    val_num = int(length * valid_perc) + train_num
    test_num = length - train_num - val_num

    set = []
    for i in range(length):
        to_add = (x_train[i].reshape(784, -1)/255, y_train[i])
        set.append(to_add)

    random.shuffle(set)
    train_set = set[0:train_num]
    valid_set = set[train_num:val_num]
    test_set = set[val_num:len(set)]

    return train_set, valid_set, test_set

def get_split_data(train_perc, x_train, y_train):
    length = len(x_train)
    train_num = int(length * train_perc)
    val_num = len(x_train)

    set = []
    for i in range(length):
        to_add = (x_train[i].reshape(784, -1)/255, y_train[i])
        set.append(to_add)

    random.shuffle(set)
    train_set = set[0:train_num]
    valid_set = set[train_num:val_num]

    return train_set, valid_set



