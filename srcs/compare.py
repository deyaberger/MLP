try:
    from utils import conf, load_eval_from_csv
    import matplotlib.pyplot as plt
    import glob
except ModuleNotFoundError as e:
    import sys
    print(f"{e}\nPlease run 'pip install -r requirements.txt'")
    sys.exit()

metric = "loss" ### Change this if you want to plot another metric
### choices = ["loss", "val_loss", "mean_sensitivity", "mean_specificity", "mean_precision", "mean_f1"]

if __name__ == "__main__":
    '''
    A very simple function to plot the evolution of the different models we have trained
    '''
    list_models = {}
    any = 0
    for name in glob.glob(f"{conf.model_folder}*"):
        if name.startswith(f"{conf.model_folder}{conf.eval_prefix}"):
            any = 1
            y = load_eval_from_csv(name, conf.eval[metric])
            x = list(range(len(y)))
            start, end = name.rfind(conf.eval_prefix), name.rfind(".csv")
            model_name = name[start:end]
            plt.plot(x, y, label = model_name)

    if any == 1:            
        plt.ylabel(metric)
        plt.xlabel('iterations')
        plt.title('Comparing models')
        plt.legend()
        plt.show()
    else:
        print(f"Nothing found in {conf.model_folder}")