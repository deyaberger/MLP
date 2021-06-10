from utils import load_file
from modules import Model, Layer


model_path = "model_13:43.pkl"

if __name__ == "__main__":
    infos = load_file(model_path)
    list_layers = []
    for l in infos["layers"]:
        layer = Layer(units = 4, activation = "softmax", input_size = X.shape[1])
    # model.compile(loss = conf.loss, optimizer = conf.optimizer)