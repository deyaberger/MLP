import pickle
from utils import conf
from modules import Model, Layer, ModelEvaluation, Plot
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    with open("matrix.pkl", "rb") as f:
        info = pickle.load(f)
    X = info["X"]
    H = info["y_predicted"]
    y = info["y"]
    
    model = Model([
        # Layer(units = 4, activation = "softmax", input_size = X.shape[1]),
        Layer(units = 6, activation = "sigmoid", input_size = X.shape[1]),
        Layer(units = 3, activation = "sigmoid"),
        Layer(units = 5, activation = "sigmoid"),
        Layer(units = 4, activation = "softmax"),
        
    ])
    model.compile(loss = conf.loss, optimizer = conf.optimizer)
    score = ModelEvaluation()
    yhat = model.fit(X, y, score)
    model.save_architecture(conf.model_path)
    model.save_weights(conf.weights_path)
    score.save(conf.eval_path)
    if conf.verbose == True:
        print(f'loss = {score.history[0][0]}')
        print(f'mean f1 = {score.history[0][-1]}')
        print("*********\n")
        print(f'loss = {score.history[-1][0]}')
        print(f'mean f1 = {score.history[-1][-1]}')
    score.history = np.array(score.history)
    if conf.graph == True:
        fig = plt.figure(1)
        plot = Plot()
        plot.show_graph("loss_evolution", list(range(score.history.shape[0])), score.history[:, 0], fig, "iterations", "loss")

        