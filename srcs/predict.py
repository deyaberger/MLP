from utils import load_json, from_json_to_layers, load_weights_from_csv
from modules import Model, Layer
import pickle
import numpy as np


architecture = "model.json"
weights_file = "weights"

if __name__ == "__main__":
    with open("matrix.pkl", "rb") as f:
        info = pickle.load(f)
    X = info["X"]
    H = info["y_predicted"]
    y = info["y"]
    
    model_architecture = load_json(architecture)
    layers_list = from_json_to_layers(model_architecture, Layer)
    model = Model(layers_list)
    model.compile(loss = model_architecture['loss'], optimizer = model_architecture['optimizer'])
    load_weights_from_csv(weights_file, model)
    prediction = model.feed_forward(X)
    loss = model.loss_function(prediction, y)
    print(loss)