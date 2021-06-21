try:
    import numpy as np
    from numpy.random import rand
    from math import sqrt
    import json
    import pickle
    import sys
except ModuleNotFoundError as e:
    import sys
    print(f"{e}\nPlease run 'pip install -r requirements.txt'")
    sys.exit()

def print_and_exit(msg):
    print(msg)
    sys.exit()


def xavier_init(input_size, units):
    '''
    Initializing the weights and bias in a "not so random way":
    we want the variance to remain the same as we pass through each layer (to avoid values to explode or vanish to zero)
    '''
    lower, upper = -(1.0 / sqrt(input_size)), (1.0 / sqrt(input_size))
    np.random.seed(70) ### Remove it if you want more random results
    weights = rand(input_size, units)
    weights = weights * (upper - lower) + lower
    return (weights)


def save_json(name, object):
    with open(name, "w") as f:
        json.dump(object, f)


def load_json(name):
    try:
        with open(name, "r") as f:
            info = json.load(f)
        return (info)
    except FileNotFoundError as e:
        print_and_exit(f"Error in function '{load_json.__name__}' :\n{e}")


def from_json_to_layers(infos, Layer):
    '''
    Reading our json dictionnary to recreate the layers list for our model
    '''
    layers_list = []
    try:
        for i, l in enumerate(infos["layers"]):
            if i == 0:
                layer = Layer(units = l["output_size"], activation = l["activation"], input_size = l["input_size"])
            else:
                layer = Layer(units = l["output_size"], activation = l["activation"])
            layers_list.append(layer)
        return (layers_list)
    except KeyError as e:
        print_and_exit(f"Error in function '{from_json_to_layers.__name__}' :\n{e}")


def load_weights_from_pickle(path, model):
    try:
        with open(path, "rb") as f:
            weights = pickle.load(f)
        for i, w in enumerate(weights):
            model.layers[i].w = w
    except FileNotFoundError as e:
        print_and_exit(f"Error in function 'load_weights_from_pickle' :\n{e}")


def load_eval_from_csv(file, col):
    eval = np.genfromtxt(file, delimiter = ",")
    return (eval[:, col])


def add_bias_units(X):
    '''
    Unused function: could be an alternative to simplify how we write our calculus
    '''
    bias_units = np.ones((X.shape[0], 1))
    X = np.concatenate((bias_units, X), axis = 1)
    return (X)