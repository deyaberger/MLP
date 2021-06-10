import numpy as np
from numpy.random import rand
from math import sqrt
import pickle


def xavier_init(input_size, units):
    lower, upper = -(1.0 / sqrt(input_size)), (1.0 / sqrt(input_size))
    np.random.seed(70) ### ! To be changed eventualy if you want to improve results
    weights = rand(input_size, units)
    weights = weights * (upper - lower) + lower
    return (weights)


def save_in_file(name, object):
    with open(name, "wb") as f:
        pickle.dump(object, f)


def load_file(name):
    with open(name, "rb") as f:
        info = pickle.load(f)
    return (info)