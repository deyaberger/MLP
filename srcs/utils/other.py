import numpy as np
from numpy.random import rand
from math import sqrt
import json
import pickle


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


def load_weights_from_pickle(path, model):
    with open(path, "rb") as f:
        weights = pickle.load(f)
    for i, w in enumerate(weights):
        model.layers[i].w = w

def load_eval_from_csv(file, col):
    eval = np.genfromtxt(file, delimiter = ",")
    return (eval[:, col])


def add_bias_units(X):
    bias_units = np.ones((X.shape[0], 1))
    X = np.concatenate((bias_units, X), axis = 1)
    return (X)
        