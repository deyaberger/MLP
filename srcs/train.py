import pickle
from utils import conf
from modules import Model, Layer, ModelEvaluation
# from datetime import datetime


if __name__ == "__main__":
    with open("matrix.pkl", "rb") as f:
        info = pickle.load(f)
    X = info["X"]
    H = info["y_predicted"]
    y = info["y"]
    
    model = Model([
        Layer(units = 4, activation = "softmax", input_size = X.shape[1])
    ])
    model.compile(loss = conf.loss, optimizer = conf.optimizer)
    score = ModelEvaluation()
    yhat = model.fit(X, y, score)
    # hour = datetime.today().strftime('%H:%M')
    # score.save(f"score_{hour}.pkl")
    model.save_architecture(f"model")
    model.save_weights(f"weights")
    
    # model.save_weights(f"weights_{hour}.pkl")

        