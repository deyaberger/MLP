import pickle
from utils import conf
from modules import Model, Layer, ModelEvaluation


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
    # score.evaluation(y, yhat)
    # yhat = model.layers[-1].a
    # yhatmax = (yhat == yhat.max(axis=1, keepdims = True)).astype(int)
    # Hmax = (H == H.max(axis=1, keepdims = True)).astype(int)
    # ymax = (y == y.max(axis=1, keepdims = True)).astype(int)
    # err = 0
    # for yhat, y in zip(yhatmax, Hmax):
    # 	if (np.argmax(yhat) != np.argmax(y)):
    # 		err += 1
    # success = round(((1 - (err / yhatmax.shape[0])) * 100), 2)
    # print(f"success =  {success}%")
        