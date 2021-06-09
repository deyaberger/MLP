import numpy as np
from numpy.random import rand
from math import sqrt


def xavier_init(input_size, units):
    lower, upper = -(1.0 / sqrt(input_size)), (1.0 / sqrt(input_size))
    np.random.seed(70) ### ! To be changed eventualy if you want to improve results
    weights = rand(input_size, units)
    weights = weights * (upper - lower) + lower
    return (weights)