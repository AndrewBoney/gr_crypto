import tensorflow as tf
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import random
import multiprocessing
from sklearn.metrics import confusion_matrix

cores = multiprocessing.cpu_count() 
tf.config.threading.set_inter_op_parallelism_threads(cores-1)

# model = tf.keras.models.load_model("models/multivar_lstm_model")
# data = np.load("data/model_data/multivar_lstm.npz", allow_pickle=True)
model = tf.keras.models.load_model("models/base_lstm_mult_model")
data = np.load("data/model_data/base_lstm_mult.npz", allow_pickle=True)
# norms = np.load("data/model_data/multivar_lstm_norms.npz")

asset_details = pd.read_csv("data/asset_details.csv")
assets = [str(i) for i in asset_details["Asset_ID"]]

test_x = data["test_x"]
test_y = data["test_y"]
train_x = data["train_x"]
train_y = data["train_y"]
time = pd.Series(pd.to_datetime(data["time"]))

# Time splits
n = len(time)
train_time = time[0:int(n*0.8)]
test_time = time[int(n*0.8):]

means = norms["means"]
sds = norms["sds"]

preds = model.predict(test_x)

def get_3d_means(arr):
    means = np.mean(arr, axis = (0, 1))
    print(means)
    return means

get_3d_means(test_x)
get_3d_means(test_y)
get_3d_means(preds)

preds_actual = (preds * sds) + means
test_x_actual = (test_x * sds) + means
test_y_actual = (test_y* sds) + means

get_3d_means(preds_actual)
get_3d_means(test_y_actual)

# Charting
def plot_pred_3d(preds, x, y, asset_index, seed):
    random.seed(seed)
    rand = random.randint(0, preds.shape[0])
    print("seed is " + str(rand))
    
    preds_compare = preds[rand, :, asset_index].reshape(-1, 1)
    x_compare = x[rand, :, asset_index].reshape(-1, 1)
    y_compare = y[rand, :, asset_index].reshape(-1, 1)
    
    actuals_plt = np.concatenate([x_compare, 
                                  y_compare])
    
    empty = np.empty((x_compare.shape[0], x_compare.shape[1]))
    empty[:] = np.nan
    preds_plt = np.concatenate([empty, preds_compare])
    
    plt.plot(actuals_plt, label = "actuals")
    plt.plot(preds_plt, label = "preds")
    plt.legend()
    plt.show()

def plot_pred_2d(preds, x, y, seed):
    random.seed(seed)
    rand = random.randint(0, preds.shape[0])
    print("seed is " + str(rand))
    
    preds_compare = preds[rand, :].reshape(-1, 1)
    x_compare = x[rand, :].reshape(-1, 1)
    y_compare = y[rand, :].reshape(-1, 1)
    
    actuals_plt = np.concatenate([x_compare, 
                                  y_compare])
    
    empty = np.empty((x_compare.shape[0], x_compare.shape[1]))
    empty[:] = np.nan
    preds_plt = np.concatenate([empty, preds_compare])
    
    plt.plot(actuals_plt, label = "actuals")
    plt.plot(preds_plt, label = "preds")
    plt.legend()
    plt.show()

for i in range(10, 20):
    plot_pred_2d(preds, test_x[:, :, 2], test_y, i)

# Somethings not working here, as a lot of the data doesn't make sense
# I think this is timestamps being wrong. Need some way to link price back 
# to time
# Might be that you can just merge it, although floating point error could
# be a problem
def make_target(data, time, details, assets):
    weights = np.array(list(details.Weight))
    
    # TODO: double check assets are in the right order
    targets = pd.DataFrame((data[:, 0, :] / data[:, 15, :]) - 1,
                           columns = assets, index = time)

    targets = targets.reindex(pd.date_range(time.min(), time.max(), 
                                         name = "time", freq = "1 min"),
                              fill_value = np.nan)
    
    targets['m'] = np.average(targets.fillna(0), axis=1, weights=weights)
    
    m = targets['m']

    num = targets.multiply(m.values, axis=0).rolling(3750).mean().values
    denom = m.multiply(m.values, axis=0).rolling(3750).mean().values
    beta = np.nan_to_num(num.T / denom, nan=0., posinf=0., neginf=0.)

    targets = targets - (beta * m.values).T
    targets.drop('m', axis=1, inplace=True)
    
    return targets
    
# Values before 3750 are nonse. Should try and fix, but short term I'm
# just going to filter out
pred_targets = make_target(preds_actual, test_time, asset_details, assets)
actual_targets = make_target(test_y_actual, test_time, asset_details, assets)


def evaluate_regression(pred, actual):
    errors = pred - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    
def evaluate_up_down(pred, actual):
    pred_up = pred > 0
    actual_up = actual > 0
    print(confusion_matrix(actual_up, pred_up))

for c in pred_targets.columns:
    print(str(asset_details[asset_details["Asset_ID"] == int(c)]["Asset_Name"]))
    evaluate_regression(pred_targets[c], actual_targets[c])
    evaluate_up_down(pred_targets[c], actual_targets[c])
    
