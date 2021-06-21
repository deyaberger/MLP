try:
    from utils import conf, load_json, from_json_to_layers, load_weights_from_pickle
    from modules import Model, Layer, Dataset, ModelEvaluation
    import argparse
    from termcolor import cprint, colored
except ModuleNotFoundError as e:
    import sys
    print(f"{e}\nPlease run 'pip install -r requirements.txt'")
    sys.exit()


def parse_arguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('-v', '--verbose', help='increase output verbosity', type=int, default=0, choices = [0, 1])
	args = parser.parse_args()
	return (args)

architecture = conf.model_path + ".json"
weights_file = conf.weights_path + ".pkl"

if __name__ == "__main__":
    args = parse_arguments()
    if args.verbose == 1:
        cprint("\n** Reading data file **", 'magenta')
    data = Dataset(conf.datafile)
    if args.verbose == 1:
        cprint("\n** Feature scaling our data to converge faster **", 'magenta')
    data.feature_scale_normalise()
    if args.verbose == 1:
        cprint(f"\n** Splitting our data into training set ({int(conf.train_size * 100)}%) and testing set ({int((1 - conf.train_size) * 100)}%) **", 'magenta')
    data.split_data()
    model_architecture = load_json(architecture)
    layers_list = from_json_to_layers(model_architecture, Layer)
    model = Model(layers_list)
    model.compile(loss = model_architecture['loss'], optimizer = model_architecture['optimizer'])
    load_weights_from_pickle(weights_file, model)
    if args.verbose == 1:
        print(colored(f"\n** Recreating model '{conf.name}' architecture: **", 'magenta') + f"\nNumber of Layers : {len(model.layers)}\nLoss : {model_architecture['loss']}\nOptimizer : {model_architecture['optimizer']}")
    prediction = model.feed_forward(data.X_test)
    loss = model.loss_function(prediction, data.y_test)
    cprint(f"\n** After making one prediction with our trained model: **", "magenta")
    print(f"loss = {loss}\n")
    score = ModelEvaluation(data.X_test, data.y_test)
    score.evaluation(prediction)
    score.keep_track(loss, loss)
    if args.verbose == 1:
        print(score)