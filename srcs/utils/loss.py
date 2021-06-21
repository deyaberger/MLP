import numpy as np
from utils.config import conf


def crossentropy(a, y):
    '''
    Exctly the same as the binary crossentropy loss, but works for multiple classifications (not only binary)
    '''
    log_a = np.log(a)
    temp_loss = np.sum((log_a * y), axis = 1, keepdims = True)
    loss = np.mean(temp_loss) * -1.0
    return (loss)

def binary_crossentropy(a, y):
    '''
    Binary crossentropy loss = - (1 / N) * Sum((y * log(a)) + ((1 - y) * log(1 - a)))
    The vectorized implementation is: (1 / N) * (-(y * log(a)) - ((1 - y)  * log(1 - a)))
    '''
    log_a = np.log(a)
    m_log_a = np.log(1 - a)
    m_y = 1 - y
    temp_loss = -(log_a * y) - (m_log_a * m_y)
    loss = np.mean(temp_loss)
    return (loss)


def crossentropy_derivative(a, y):
    d_cross = y / (a + conf.epsilon)
    d_cross = (d_cross / y.shape[0]) * -1.0
    return (d_cross)


def mse(a, y):
    '''
    Mean squared error
    '''
    loss = np.sum(np.square(a - y), axis = 1, keepdims=True)
    loss = np.mean(loss) / 2
    return (loss)

    
def mse_derivative(a, y):
    d_mse = (a - y)
    return (d_mse)


def get_loss(loss_name):
    loss_function = None
    if loss_name == "binary_crossentropy":
        loss_function, loss_function_derivative = binary_crossentropy, crossentropy_derivative
    elif loss_name == "crossentropy":
        loss_function, loss_function_derivative = crossentropy, crossentropy_derivative
    elif loss_name == "mse":
        loss_function, loss_function_derivative = mse, mse_derivative
    return loss_function, loss_function_derivative