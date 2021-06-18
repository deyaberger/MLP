from utils import conf, conf
from modules import Model, Layer, ModelEvaluation, Plot, Dataset
import numpy as np
import matplotlib.pyplot as plt


if __name__ == "__main__":
    data = Dataset(conf.datafile)
    data.feature_scale_normalise()
    data.split_data()
    print(data)
    model = Model([
        Layer(units = 6, activation = "sigmoid", input_size = data.X.shape[1]),
        # Layer(units = 3, activation = "sigmoid"),
        # Layer(units = 5, activation = "sigmoid"),
        Layer(units = data.y.shape[1], activation = "softmax"),
        
    ])
    model.compile(loss = conf.loss, optimizer = conf.optimizer)
    score = ModelEvaluation(data.X_test, data.y_test)
    yhat = model.fit(data.X_train, data.y_train, score)
    model.save_architecture(conf.model_path)
    model.save_weights(conf.weights_path)
    score.save(conf.eval_path)
    if conf.verbose == True:
        score.print_summary()
    score.history = np.array(score.history)
    if conf.graph == True:
        fig = plt.figure(1)
        plot = Plot()
        plot.show_graph("loss_evolution", list(range(score.history.shape[0])), score.history[:, 0], fig, "iterations", "loss")

        