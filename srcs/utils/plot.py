from utils.config import conf
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    import sys
    print(f"{e}\nPlease run 'pip install -r requirements.txt'")
    sys.exit()

    
def plot_metrics(metrics, history):
    '''
    Ploting different metrics on the same figure
    '''
    for metric in metrics:
        x = list(range(history.shape[0]))
        y = history[:, conf.eval[metric]]
        plt.plot(x, y, label = metric)
    plt.title('Evolution of validation metrics')
    plt.xlabel('iterations')
    plt.ylabel('metric value')
    plt.legend()
    plt.show()
