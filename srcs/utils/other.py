import numpy as np
from numpy.random import rand
from math import sqrt
import json
import matplotlib.pyplot as plt


def xavier_init(input_size, units):
    lower, upper = -(1.0 / sqrt(input_size)), (1.0 / sqrt(input_size))
    np.random.seed(70) ### ! To be changed eventualy if you want to improve results
    weights = rand(input_size, units)
    weights = weights * (upper - lower) + lower
    return (weights)


def save_json(name, object):
    with open(name, "w") as f:
        json.dump(object, f)


def load_json(name):
    with open(name, "r") as f:
        info = json.load(f)
    return (info)


def from_json_to_layers(infos, Layer):
    layers_list = []
    for i, l in enumerate(infos["layers"]):
        if i == 0:
            layer = Layer(units = l["output_size"], activation = l["activation"], input_size = l["input_size"])
        else:
            layer = Layer(units = l["output_size"], activation = l["activation"])
        layers_list.append(layer)
    return (layers_list)


def load_weights_from_csv(csv_prefix, model):
    for i, layer in enumerate(model.layers):
        weights = np.genfromtxt(f"{csv_prefix}_{i}.csv", delimiter = ",")
        layer.w = weights
        

def show_graph(title, x, y, fig, x_label, y_label):
    plt.figure(fig)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.plot(x, y)
    plt.show()