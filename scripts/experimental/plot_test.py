import tensorflow as tf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
from sklearn.metrics import confusion_matrix

cores = multiprocessing.cpu_count() 
tf.config.threading.set_inter_op_parallelism_threads(cores-1)

model_name = "return_lstm"

model = tf.keras.models.load_model("models/" + model_name)
data = np.load("data/model_data/" + model_name + ".npz", allow_pickle=True)
# norms = np.load("data/model_data/multivar_lstm_norms.npz")

asset_details = pd.read_csv("data/asset_details.csv")
assets = [str(i) for i in asset_details["Asset_ID"]]

# It's hard to plot with 2 step model, as you don't have the full post 15 min
# behavior

test_x = data["test_x"]
test_y = data["test_y"]
train_x = data["train_x"]
train_y = data["train_y"]
time = pd.Series(pd.to_datetime(data["time"]))

preds = model.predict(test_x)

for i in range(16):
    plt.hist(preds[:, i], bins = 50)
    plt.show()

# This is just giving static predictions, regardless of data

def plot_pred_2d(preds, x, y, seed, step = False):
    random.seed(seed)
    rand = random.randint(0, preds.shape[0])
    print("row is " + str(rand))
    
    preds_compare = preds[rand, :].reshape(-1, 1)
    x_compare = x[rand, :].reshape(-1, 1)
    y_compare = y[rand, :].reshape(-1, 1)
    
    if step:
        empty = np.empty((15, y_compare.shape[1]))
        empty[:] = np.nan
        empty[0, :] = y_compare[0, :]
        empty[14, :] = y_compare[1, :]
        y_compare = empty
        
        empty = np.empty((15, y_compare.shape[1]))
        empty[:] = np.nan
        empty[0, :] = preds_compare[0, :]
        empty[14, :] = preds_compare [1, :]
        preds_compare = empty
        
    actuals_plt = np.concatenate([x_compare, 
                                  y_compare])
    
    empty = np.empty((x_compare.shape[0], x_compare.shape[1]))
    empty[:] = np.nan
    preds_plt = np.concatenate([empty, preds_compare])
    
    
    plt.plot(actuals_plt, label = "actuals")
    plt.scatter(range(preds_plt.shape[0]), preds_plt, 
                label = "preds", color = "orange")
    plt.scatter(range(preds_plt.shape[0]), actuals_plt, 
                color = "blue")
    plt.legend()
    plt.show()

for i in range(5, 10):
    # plot_pred_2d(preds, test_x, test_y, i, step = True)
    plot_pred_2d(preds, test_x, test_y, i, step = False)

plot_pred_2d(preds, test_x, test_y, 10)
plot_pred_2d(preds, test_x, test_y, 11)
plot_pred_2d(preds, test_x, test_y, 100)




