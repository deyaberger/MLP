from utils import conf, plot_metrics
from modules import Model, Layer, ModelEvaluation, Dataset
import numpy as np
import argparse
from termcolor import cprint, colored


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--loss', help='Choose your loss function', type=str, default="binary_crossentropy", choices = ["binary_crossentropy", "crossentropy", "mse"])
    parser.add_argument('-o', '--optimizer', help='Choose your optimizer', type=str, default="gradient_descent", choices = ["gradient_descent", "momentum"])
    parser.add_argument('-v', '--verbose', help='increase output verbosity', type=int, default=0, choices = [0, 1])
    parser.add_argument('-ea', '--early_stop', help='Helps avoiding overfitting', action = 'store_true')
    parser.add_argument('-p', '--plot', help='Display evolution of validation metrics', type = str, nargs='+', choices = ["loss", "val_loss", "mean_sensitivity", "mean_specificity", "mean_precision", "mean_f1"])
    args = parser.parse_args()
    return (args)

def init_model(args):
    model = Model([
        Layer(units = 6, activation = "sigmoid", input_size = data.X.shape[1]),
        Layer(units = 3, activation = "sigmoid"),
        Layer(units = 5, activation = "sigmoid"),
        Layer(units = data.y.shape[1], activation = "softmax"),
    ])
    model.compile(loss = args.loss, optimizer = args.optimizer)
    model.args = args
    if args.verbose == 1:
        print(colored(f"\n** Model has been compiled with the following architecture: **", 'magenta') + f"\nNumber of Layers : {len(model.layers)}\nLoss : {args.loss}\nOptimizer : {args.optimizer}")
    return (model)

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
    model = init_model(args)
    score = ModelEvaluation(data.X_test, data.y_test)
    if args.verbose == 1:
        cprint(f"\n** Training our model for {conf.epochs} epochs **", "magenta")
    model.fit(data.X_train, data.y_train, score)
    if args.verbose == 1:
        score.print_summary()
    model.save_architecture(conf.model_path)
    model.save_weights(conf.weights_path)
    score.save(conf.eval_path)
    if args.verbose == 1:
        print(colored(f"\n** The following files are saved in the folder '{conf.model_folder}' : **", "magenta") + f"\nModel's architecture: {conf.model_prefix}{conf.name}.json\nScore evolution : {conf.eval_prefix}{conf.name}.csv\nWeights for each layer : {conf.weights_prefix}{conf.name}.pkl\n")
    if args.plot:
        plot_metrics(args.plot, np.array(score.history))        