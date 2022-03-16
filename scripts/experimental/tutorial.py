# Working through tutorial in 
# https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import ConvLSTM2D
from tensorflow.keras.layers import RepeatVector
from tensorflow.keras.layers import TimeDistributed
import tensorflow as tf
from time import time
import multiprocessing
import random

cores = multiprocessing.cpu_count() 
tf.config.threading.set_inter_op_parallelism_threads(cores-1)

root_folder = "data"

wide_close = pd.read_csv(root_folder + "/working/wide_close.csv")
wide_target = pd.read_csv(root_folder + "/working/wide_target.csv")
asset_details = pd.read_csv(root_folder + "/asset_details.csv")

assets = [str(i) for i in asset_details["Asset_ID"]]

# Getting target equivalent. This is to allow us to make multi-output 
# predictions (i.e minute by minute, for 1:15 mins post x)
# I'll define a DataFrame with "Target" and "Close"
# keep = ["time", "1"]
# assert_target = wide_close[keep].merge(wide_target[keep], on = "time",
#                                       suffixes = [".close", ".target"])

# Defining functions for running for model conditions, giving us 
# flexibility. 

# Models
## Simple LSTM
def simple_lstm(x_steps, y_steps, n_features):
    model = Sequential([
        LSTM(10, activation = "relu", input_shape = (x_steps, n_features)), 
        Dense(y_steps)
    ])
    return model

## Med LSTM
def med_lstm(x_steps, y_steps, n_features_x):
    model = Sequential([
        LSTM(50, activation = "relu", return_sequences = True,
             input_shape = (x_steps, n_features), dropout = 0.4),
        LSTM(20, activation = "relu"),
        Dense(y_steps)
    ])
    return model

# Encoder / Decoder LSTM model
def end_dec_lstm(x_steps, y_steps, n_features_x, n_features_y):
    model = Sequential([
        LSTM(200, activation = "relu", input_shape = (x_steps, n_features_x)), 
        RepeatVector(y_steps),
        LSTM(100, activation = "relu", return_sequences = True), 
        TimeDistributed(Dense(n_features_y))    
    ])
    return model

# Compiler
def base_compiler(model):
    return model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(learning_rate = 0.0005),
                    metrics=[tf.metrics.MeanAbsoluteError()])

def cosine_compiler(model):
    return model.compile(loss=tf.keras.losses.CosineSimilarity(axis=1),
                    optimizer=tf.optimizers.Adam(learning_rate = 0.0005),
                    metrics=[tf.metrics.MeanAbsoluteError()])

    

wide_close[assets] = wide_close[assets].replace([np.inf, -np.inf], np.nan)

def full_model(data_in, col_in, col_out, x_steps, y_steps, fun, compiler,
               save_loc, save = True, epochs = 50, patience = 2):
    
    n_features_x = len(col_in)
    n_features_y = len(col_out)
    
    # TODO: CHeck length of sequences are equal
    x_sequences = np.array(data_in[col_in]).reshape(-1, n_features_x)
    y_sequences = np.array(data_in[col_out]).reshape(-1, n_features_y)
    
    # set index measures. 
    if isinstance(x_steps, int):
        ix_in = list(range(x_steps))
    elif isinstance(x_steps, list):
        ix_in = x_steps
    else:
        raise TypeError("x_steps should be either an int or list")
        
    if isinstance(y_steps, int):
        ix_out = list(range(1, y_steps + 1))
    elif isinstance(y_steps, list):
        ix_out = y_steps
    else:
        raise TypeError("y_steps should be either an int or list")        

    def split_sequences(x_sequences, y_sequences, ix_in, ix_out):
        X, y = list(), list()
        for i in range(max(ix_in), len(x_sequences) - max(ix_out)):
            end_ix = [i - x for x in ix_in]
            out_end_ix = [i + x for x in ix_out]
            
            seq_x, seq_y = x_sequences[end_ix, :], y_sequences[out_end_ix, :]
            X.append(seq_x)
            y.append(seq_y)
        return np.array(X), np.array(y)
    
    t0 = time()
    all_x, all_y = split_sequences(x_sequences, y_sequences, ix_in, ix_out)
    t1 = time()
    print("split_sequences took " + str(round(t1 - t0)) + " seconds")
    
    # Remove NAs
    def remove_nas_3d(x, y):
        missing_x = ~np.isnan(x).any(axis=1)
        missing_row_x = ~np.logical_not(missing_x).any(axis=1)
        
        missing_y = ~np.isnan(y).any(axis=1)
        missing_row_y = ~np.logical_not(missing_y).any(axis=1)
        
        both = np.logical_and(missing_row_x, missing_row_y)
        
        return x[both, :, :], y[both, :, :], both # also return indexes
    
    t0 = time()
    filt_x, filt_y, both = \
        remove_nas_3d(all_x.reshape(all_x.shape[0], all_x.shape[1], n_features_x), 
                      all_y.reshape(all_y.shape[0], all_y.shape[1], n_features_y))
    t1 = time()
    print("remove_nas_3d took " + str(round(t1 - t0)) + " seconds")
    
    # Saving timestamps
    time_filt = np.array(data_in["time"][max(ix_in):(len(data_in)-max(ix_out))][both])
    
    # Train / Test Splits
    # TODO: Spit into folds for CV
    n = len(filt_y)
    train_x = filt_x[0:int(n*0.8),:,:]
    train_y = filt_y[0:int(n*0.8),:,:]
    
    test_x = filt_x[int(n*0.8):, :,:]
    test_y = filt_y[int(n*0.8):,:,:]
    
    print("train_x is ", train_x.shape)
    print("train_y is ", train_y.shape)
    print("test_x is ", test_x.shape)
    print("test_y is ", test_y.shape)
    
    # Create model, from predefined fun. Allows for flexibility 
    # (although might make saving hard)
    del([all_x, all_y, filt_x, filt_y, x_sequences, y_sequences])    
    
    # Get Model, specified in fun
    x_len = len(ix_in)
    y_len = len(ix_out)
    model = fun(x_len, y_len, n_features_x)    
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
    
    # Compile model with compiler
    compiler(model)
    
    model.summary()
    
    history = model.fit(train_x, train_y, epochs = epochs, batch_size = 32,
                        validation_split = 0.1, callbacks = [early_stopping])

    def plot_loss(history):
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        # plt.ylim([0, 10])
        # plt.xlabel('Epoch') 
        # plt.ylabel('Error [MPG]')
        plt.legend()
        plt.grid(True)
        
    plot_loss(history)    
    
    if save == True:
        np.savez(save_loc, test_x = test_x, test_y = test_y, 
                 train_x = train_x, train_y = train_y, time = time_filt)
        
    return model, history

base_lstm_model, base_lstm_history = \
    full_model(wide_close, "1", "1", 60, 15, 1, simple_lstm, base_compiler)

base_lstm_mult_model, base_lstm_mult_history = \
    full_model(data_in = wide_close, col_in = assets, col_out = "1", 
               x_steps = 60, y_steps = 15, fun = simple_lstm_mult, 
               compiler = base_compiler, 
               save_loc = "data/model_data/base_lstm_mult.npz")
    
# !mkdir "models"
# !mkdir "models\base_lstm_model"
# mkdir "models\base_lstm_mult_model"
base_lstm_model.save("models/base_lstm_model")
base_lstm_mult_model.save("models/base_lstm_mult_model")

# 2 step
step_base_lstm_model, step_base_lstm_history = \
    full_model(data_in = wide_close, col_in = "1", col_out = "1", 
           x_steps = 60, y_steps = [1, 16], fun = simple_lstm, 
           compiler = base_compiler, 
           save_loc = "data/model_data/step_lstm.npz")

!mkdir "models/step_lstm"
step_base_lstm_model.save("models/step_lstm")

# Return Based
def log_return(series, periods=1):
    return np.log(series).diff(periods=periods)

close_returns = wide_close[assets].apply(log_return)
close_returns["time"] = wide_close["time"]
close_returns[assets] = close_returns[assets].replace([np.inf,-np.inf],np.nan)

import seaborn as sns
long = close_returns[assets].melt().dropna()
sns.violinplot(x = "variable", y = "value", data = long)

simple_return_model, simple_return_history = \
    full_model(data_in = close_returns, col_in = "1", col_out = "1", 
           x_steps = 60, y_steps = 16, fun = simple_lstm, 
           compiler = base_compiler, 
           save_loc = "data/model_data/return_lstm.npz")

!mkdir "models/return_lstm"
simple_return_model.save("models/return_lstm")

med_return_model, med_return_history = \
    full_model(data_in = close_returns, col_in = "1", col_out = "1", 
           x_steps = 60, y_steps = [1, 16], fun = med_lstm, 
           compiler = base_compiler, 
           save_loc = "data/model_data/med_return_lstm.npz")

# Normalised all vars
means  = wide_close[assets].mean()
sds = wide_close[assets].std()

norm = (wide_close[assets] - means) / sds
norm["time"] = wide_close["time"]

multivar_lstm_model, multivar_lstm_history = \
    full_model(data_in = norm, col = assets, x_steps = 60, y_steps = 16, 
               fun = end_dec_lstm, compiler = base_compiler, 
               save_loc = "data/model_data/multivar_lstm.npz")

np.savez("data/model_data/multivar_lstm_norms.npz", means = means, sds = sds)

# Save
!mkdir "models/multivar_lstm_model"
multivar_lstm_model.save("models/multivar_lstm_model")


# Make explainer file
explain = pd.DataFrame({"model": ["base_lstm_model", "multivar_lstm_model"],
                        "assets": ["bitcoin", "all"],
                        "x_steps": [60, 15],
                        "x_how": ["wide_close, raw", "wide_close, normalised"],
                        "y_how": ["15 steps", "15 steps"],
                        "description": ["Simple LSTM with 10 neurons", 
                                        "2 LSTM layers with 200/100 neurons, and encoder / decoder layer"]})


# Plot


