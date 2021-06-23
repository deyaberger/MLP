try:
    from utils import conf, load_eval_from_csv
    import matplotlib.pyplot as plt
    import glob
    import argparse
except ModuleNotFoundError as e:
    import sys
    print(f"{e}\nPlease run 'pip install -r requirements.txt'")
    sys.exit()

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--metric', help='Choose metric to compare', type=str, default="loss", choices = ["loss", "val_loss", "mean_sensitivity", "mean_specificity", "mean_precision", "mean_f1"])
    args = parser.parse_args()
    return (args)

if __name__ == "__main__":
    '''
    A very simple function to plot the evolution of the different models we have trained
    '''
    list_models = {}
    any = 0
    args = parse_arguments()
    for name in glob.glob(f"{conf.model_folder}*"):
        if name.startswith(f"{conf.model_folder}{conf.eval_prefix}"):
            any = 1
            y = load_eval_from_csv(name, conf.eval[args.metric])
            x = list(range(len(y)))
            start, end = name.rfind(conf.eval_prefix), name.rfind(".csv")
            model_name = name[start:end]
            plt.plot(x, y, label = model_name)

    if any == 1:            
        plt.ylabel(args.metric)
        plt.xlabel('iterations')
        plt.title('Comparing models')
        plt.legend()
        plt.show()
    else:
        print(f"Nothing found in {conf.model_folder}")