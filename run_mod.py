"""
Setup:
    - Import Libraries
    - Setup tf on multiple cores
    - Import Data
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns

from time import time
import multiprocessing
import random
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, ConvLSTM2D, Flatten

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from joblib import dump, load

from mod.prep import log_return, log_return_np, preprocess
from mod.model import return_pred
from mod.eval import evaluate_regression, evaluate_up_down

cores = multiprocessing.cpu_count() 
tf.config.threading.set_inter_op_parallelism_threads(cores-1)

root_folder = "data"

wide_close = pd.read_csv(root_folder + "/working/wide_close.csv")
wide_target = pd.read_csv(root_folder + "/working/wide_target.csv")
asset_details = pd.read_csv(root_folder + "/asset_details.csv")

assets = [str(i) for i in asset_details["Asset_ID"]]

"""
Preprocess
"""

close_returns = wide_close[assets].apply(log_return)
close_returns["time"] = wide_close["time"]
close_returns[assets] = close_returns[assets].replace([np.inf,-np.inf],np.nan)

"""
Linear Regression
"""
x_steps, y_steps = 60, [1, 15]
col_in, col_out = "1", "1"
 
train_x, test_x, train_y, test_y, time_d = preprocess(data_in = wide_close, col_in, 
                                          col_out, time_col="time", x_steps, y_steps)

# 1 step
lr_1 = LinearRegression()
lr_1.fit(train_x.reshape(-1, x_steps), train_y[:,0,:].reshape(-1, 1))

true, pred = return_pred(test_x, test_y[:,0,:], lr_1)

evaluate_regression(true, pred)
evaluate_up_down(true, pred)

# 15 step
lr_15 = LinearRegression()
lr_15.fit(train_x.reshape(-1, x_steps), train_y[:,1,:].reshape(-1, 1))

true, pred = return_pred(test_x, test_y[:,1,:], lr_1)

evaluate_regression(true, pred)
evaluate_up_down(true, pred)




"""
calculate and store components seperately
process:
- first, get rolling values for each timestamp
- then, predict 1 and 15 gaps and store in array

"""

# Production

"""
Steps:
    - Get train, val test and test indices. Importantly, this 
      needs to cover all assets (even though not all assets exist)
      for the whole time period.
    - Build models

"""

assets = list(asset_details["Asset_ID"].astype(str))

# Get indexes
i = np.select(
    [
     (wide_close.index >= 0) & (wide_close.index <= (len(wide_close)*0.7)),
     (wide_close.index > (len(wide_close)*0.7)) & (wide_close.index <= (len(wide_close)*0.8))
    ],
    ["train", "val"],
    default = "test")

indexes = pd.DataFrame({"time":wide_close["time"],
                        "set":i})

for a in assets:
    print("asset", a)
    filt = indexes["set"][~pd.isna(wide_close[a])]
    counts = filt.value_counts()
    df = pd.DataFrame({"counts":counts,
                        "pct":counts/np.sum(counts)})
    print(df, "\n\n")

indexes_d = {}
for s in indexes["set"].unique():
    indexes_d[s] = indexes["time"][indexes["set"] == s]

mkdir "model_files"
mkdir "model_files/linear_regression"

for a in assets:
    print("Asset", a)
    x_steps, y_steps = 60, [1, 16]
    cols_in, cols_out = a, a
     
    train_x, test_x, train_y, test_y, time_d = preprocess(wide_close, cols_in, 
                                              cols_out, "time", x_steps, y_steps)
    
    # 1 step
    lr_1 = LinearRegression()
    lr_1.fit(train_x.reshape(-1, x_steps), train_y[:,0,:].reshape(-1, 1))
    
    true, pred = return_pred(test_x, test_y[:,0,:], lr_1)

    print("Model 1 Metrics")
    evaluate_regression(true, pred)
    evaluate_up_down(true, pred)
    
    # 16 step
    lr_16 = LinearRegression()
    lr_16.fit(train_x.reshape(-1, x_steps), train_y[:,1,:].reshape(-1, 1))

    true, pred = return_pred(test_x, test_y[:,1,:], lr_16)

    print("Model 16 Metrics")
    evaluate_regression(true, pred)
    evaluate_up_down(true, pred)

    dump(lr_1, f"model_files/linear_regression/lr_{a}_1")
    dump(lr_16, f"model_files/linear_regression/lr_{a}_16")
    dump(time_d, "model_files/linear_regression/lr_times")


"""
Random Forest
"""

rf = RandomForestRegressor(n_jobs=-1)
# start = time.time()
rf.fit(train_x.reshape(-1, x_steps), train_y.reshape(-1))
# print("Took:", round(start-time.time()))

