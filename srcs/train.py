import pickle
from utils import conf, show_graph
from modules import Model, Layer, ModelEvaluation
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    with open("matrix.pkl", "rb") as f:
        info = pickle.load(f)
    X = info["X"]
    H = info["y_predicted"]
    y = info["y"]
    
    model = Model([
        # Layer(units = 4, activation = "softmax", input_size = X.shape[1]),
        Layer(units = 9, activation = "softmax", input_size = X.shape[1]),
        Layer(units = 8, activation = "softmax"),
        Layer(units = 7, activation = "softmax"),
        Layer(units = 4, activation = "softmax"),
        
    ])
    model.compile(loss = conf.loss, optimizer = conf.optimizer)
    score = ModelEvaluation()
    yhat = model.fit(X, y, score)
    model.save_architecture(conf.model_name)
    model.save_weights(conf.weights_name)
    print(f'loss = {score.history[0][0]}')
    print(f'mean f1 = {score.history[0][-1]}')
    print("*********\n")
    print(f'loss = {score.history[-1][0]}')
    print(f'mean f1 = {score.history[-1][-1]}')
    fig = plt.figure(1)
    score.history = np.array(score.history)
    show_graph("loss_evolution", list(range(score.history.shape[0])), score.history[:, 0], fig, "iterations", "loss")

        