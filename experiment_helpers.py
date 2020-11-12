import MLP_3 as mlp
import helper_functions as act_func
import dataMethods as data
import layer
from datetime import datetime
import matplotlib.pyplot as plt

def get_start_data(data_split, val_split = -1):
    train_set, valid_set, test_set = data.get_data_from_files_with_vaildation_data(data_split, val_split)
    return train_set, valid_set, test_set

def create_network(neur_in_layers, activation_function, min_w = 0, max_w = 0.1):
    network = mlp.network()
    for i in range(len(neur_in_layers)-2):
        l = layer.Layer(neur_in_layers[i], neur_in_layers[i+1], activation_function, min_w, max_w)
        network.add_layer(l)
    l = layer.Layer(neur_in_layers[-2], neur_in_layers[-1], act_func.linear, min_w, max_w)
    network.add_layer(l)
    return network

def matricesForTexTable(tabOfMatrices):
    text = ''
    for i in range(len(tabOfMatrices[0])):
        start = True
        for tab in tabOfMatrices:
            if (start == True):
                text += str(tab[i])
                start = False
            else:
                text += ' & ' + str(round(tab[i], 4))
        text += ' \\\\\n'
    return text

def save_text_to_file(filename, text, title = "", side_parameters =""):
    f = open(filename, "a+")
    datetimeStr = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    f.write(datetimeStr+'\n')
    f.write(title+'\n')
    f.write(side_parameters)
    f.write(text)
    f.close()


def make_params_list(network_layers='', activation_fun = '', learning_rate = '', batch_size='', train_set_size = '', valid_set_size='', test_set_size=''):
    text = ''
    if (network_layers != ''):
        text += 'Architektura sieci: ' +  str(network_layers) + '\n'
    if (activation_fun != ''):
        text += 'Funkcja aktywacji: ' + act_func.activationString(activation_fun) + '\n'
    if (learning_rate != ''):
        text += 'Współczynnik uczenia: ' +  str(learning_rate) + '\n'
    if (batch_size != ''):
        text += 'Rozmiar paczki: ' + str(batch_size) + '\n'
    if (train_set_size != ''):
        text += 'Rozmiar zbioru treningowego: ' + str(train_set_size) + '\n'
    if (valid_set_size != ''):
        text += 'Rozmiar zbioru walidacyjnego: ' + str(valid_set_size) + '\n'
    if (test_set_size != ''):
        text += 'Rozmiar zbioru testowego: ' + str(test_set_size) + '\n'
    return text

def weights_list_to_string(wl):
    w_s = []
    for w in wl:
        w_s.append(str(w))
    return w_s