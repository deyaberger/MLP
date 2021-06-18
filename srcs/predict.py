from utils import load_json, from_json_to_layers, load_weights_from_pickle, conf
from modules import Model, Layer, Dataset
import pickle
import numpy as np


architecture = conf.model_path + ".json"
weights_file = conf.weights_path + ".pkl"

if __name__ == "__main__":
    data = Dataset(conf.datafile)
    data.feature_scale_normalise()
    data.split_data()
    model_architecture = load_json(architecture)
    layers_list = from_json_to_layers(model_architecture, Layer)
    model = Model(layers_list)
    model.compile(loss = model_architecture['loss'], optimizer = model_architecture['optimizer'])
    load_weights_from_pickle(weights_file, model)
    prediction = model.feed_forward(data.X_test)
    loss = model.loss_function(prediction, data.y_test)
    print(loss)