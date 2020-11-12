import keras
import random

def get_mnist():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
    return (x_train, y_train), (x_test, y_test)


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



