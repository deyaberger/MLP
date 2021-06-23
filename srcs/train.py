try:
    from utils import conf, plot_metrics, print_and_exit
    from modules import Model, Layer, ModelEvaluation, Dataset
    import numpy as np
    import argparse
    from termcolor import cprint, colored
except ModuleNotFoundError as e:
    import sys
    print(f"{e}\nPlease run 'pip install -r requirements.txt'")
    sys.exit()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', help='Choose the number of epochs for training', type=int, default=4000)
    parser.add_argument('-l', '--loss', help='Choose your loss function', type=str, default="binary_crossentropy", choices = ["binary_crossentropy", "crossentropy", "mse"])
    parser.add_argument('-o', '--optimizer', help='Choose your optimizer', type=str, default="gradient_descent", choices = ["gradient_descent", "momentum"])
    parser.add_argument('-v', '--verbose', help='increase output verbosity', type=int, default=0, choices = [0, 1])
    parser.add_argument('-ea', '--early_stop', help='Helps avoiding overfitting', action='store_true')
    parser.add_argument('-b', '--batch', help='Change training batch for each epoch', action='store_true')
    parser.add_argument('-p', '--plot', help='Display evolution of validation metrics', type=str, nargs='+', choices = ["loss", "val_loss", "mean_sensitivity", "mean_specificity", "mean_precision", "mean_f1"])
    parser.add_argument('-n', '--name', help='Choose model name', type=str, default="my_model")
    args = parser.parse_args()
    return (args)

def init_model(args):
    '''
    The more layers you put, the longer it will take to compute, so consider changing the number of epochs
    choices of activation functions : ["sigmoid", "softmax", "identity"] 
    '''
    model = Model([
        Layer(units = 6, activation = "sigmoid", input_size = data.X.shape[1]),
        Layer(units = 3, activation = "sigmoid"),
        Layer(units = 5, activation = "sigmoid"),
        Layer(units = data.y.shape[1], activation = "softmax"),
    ])
    model.compile(loss = args.loss, optimizer = args.optimizer)
    model.args = args
    if args.epochs < 1 or args.epochs > 30000:
        print_and_exit("Please enter a number of epochs superior to 1 and inferior to 30 000 (to avoid very long loops and overfitting)")
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
        cprint(f"\n** Training our model for {model.args.epochs} epochs **", "magenta")
    model.fit(data.X_train, data.y_train, score)
    if args.verbose == 1:
        score.print_summary()
    model.save_topology(conf.model_path + model.args.name)
    model.save_weights(conf.weights_path + model.args.name)
    score.save(conf.eval_path + model.args.name)
    if args.verbose == 1:
        print(colored(f"\n** The following files are saved in the folder '{conf.model_folder}' : **", "magenta") + f"\nModel's topology: {conf.topo_prefix}{model.args.name}.json\nScore evolution : {conf.eval_prefix}{model.args.name}.csv\nWeights for each layer : {conf.weights_prefix}{model.args.name}.pkl\n")
    if args.plot:
        plot_metrics(args.plot, np.array(score.history))        