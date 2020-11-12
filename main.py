import MLP as mlp
import danse_layer as dl
import numpy as np
import  dataMethods as data
import keras
import warnings
import layer
import MLP_3 as mlp3
#import MLP_2



#x = np.array([1, 2, 3, 4, 5]).reshape(1, -1).transpose()

#der = mlp.activationDerivative(mlp.relu)
# #print(dense1.forward(x))
#
# network = mlp.MLP()
# network.addLayer(dense1)
# network.addLayer(dense2)
#
# print(network.predict(x))

# warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
#
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
#
# trainset = data.get_train_data(50000, x_train, y_train)


# dense1 = dl.DenseLayer(784, 400, mlp.sigmoid)
# dense2 = dl.DenseLayer(400, 200, mlp.sigmoid)
# dense3 = dl.DenseLayer(200, 100, mlp.sigmoid)
# dense4 = dl.DenseLayer(100, 10, mlp.linear)
#
#
# network = mlp.MLP()
# network.addLayer(dense1)
# network.addLayer(dense2)
# network.addLayer(dense3)
# network.addLayer(dense4)
# network.train_network_with_minibatch(trainset)

# l1 = layer.Layer(784, 200, mlp.relu)
# l2 = layer.Layer(200, 100, mlp.relu)
# l3 = layer.Layer(100, 10, mlp.linear)
#
# network = mlp3.network()
# network.add_layer(l1)
# network.add_layer(l2)
# network.add_layer(l3)
#
# network.train_with_batch(trainset)

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path="mnist.npz")
train_set, valid_set = data.get_split_data(0.4, x_train, y_train)

l1 = layer.Layer(784, 200, mlp.tanh)
l2 = layer.Layer(200, 100, mlp.tanh)
l3 = layer.Layer(100, 10, mlp.linear)

network = mlp3.network()
network.add_layer(l1)
network.add_layer(l2)
network.add_layer(l3)

network.train_with_batch(train_set, valid_set)
accuracy, loss = network.test_network(valid_set)
print(accuracy)
print(loss)


