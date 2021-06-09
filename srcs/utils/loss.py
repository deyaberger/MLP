import numpy as np
from utils.config import conf


def crossentropy(a, y):
	log_a = np.log(a)
	temp_loss = np.sum((log_a * y), axis = 1, keepdims = True)
	loss = np.mean(temp_loss) * -1.0
	return (loss)


def crossentropy_derivative(a, y):
	d_cross = -1.0 * (y / (a + conf.epsilon))
	d_cross = d_cross / y.shape[0]
	return (d_cross)


def get_loss(loss_name):
	loss_function = None
	if loss_name == "crossentropy":
		loss_function, loss_function_derivative = crossentropy, crossentropy_derivative
	return loss_function, loss_function_derivative