import matplotlib.pyplot as plt
from utils.config import conf

    
def plot_metrics(metrics, history):
    for metric in metrics:
        x = list(range(history.shape[0]))
        y = history[:, conf.eval[metric]]
        plt.plot(x, y, label = metric)
    plt.title('Evolution of validation metrics')
    plt.xlabel('iterations')
    plt.ylabel('metric value')
    plt.legend()
    plt.show()
