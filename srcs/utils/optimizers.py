from utils.config import conf

def gradient_descent(layer):
	layer.w = layer.w - (conf.lr * layer.djdw)
	layer.b = layer.b - (conf.lr * layer.djdb)

def momentum(layer):
	layer.vw = conf.mm * (layer.vw) + (1 - conf.mm) * layer.djdw
	layer.vb = conf.mm * (layer.vb) + (1 - conf.mm) * layer.djdb
	layer.w = layer.w - (conf.lr * layer.vw)
	layer.b = layer.b - (conf.lr * layer.vb)


def optimizer_function(optimizer_name):
	optimizer_function = None
	if optimizer_name == "gradient_descent":
		optimizer_function = gradient_descent
	elif optimizer_name == "momentum":
		optimizer_function = momentum
	return (optimizer_function)
