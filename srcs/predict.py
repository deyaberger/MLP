from utils import conf, load_json, from_json_to_layers, load_weights_from_pickle
from modules import Model, Layer, Dataset, ModelEvaluation
import argparse

def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--verbose', help='increase output verbosity', type=int, default=0, choices = [0, 1])
	args = parser.parse_args()
	return (args)

architecture = conf.model_path + ".json"
weights_file = conf.weights_path + ".pkl"

if __name__ == "__main__":
    args = parse_arguments()
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
    score = ModelEvaluation(data.X_test, data.y_test)
    score.evaluation(prediction)
    score.keep_track(loss, loss)
    print(score)