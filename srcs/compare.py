from utils import conf, load_eval_from_csv
import matplotlib.pyplot as plt
import glob


if __name__ == "__main__":
    list_models = {}
    for name in glob.glob(f"{conf.model_folder}*"):
        if name.startswith(f"{conf.model_folder}{conf.eval_prefix}"):
            print(name)
            y = load_eval_from_csv(name, conf.eval["loss"])
            x = list(range(len(y)))
            start, end = name.rfind(conf.eval_prefix), name.rfind(".csv")
            model_name = name[start:end]
            plt.plot(x, y, label = model_name)
            
plt.ylabel('loss')
plt.xlabel('iterations')
plt.title('Comparing models')
plt.legend()
plt.show()