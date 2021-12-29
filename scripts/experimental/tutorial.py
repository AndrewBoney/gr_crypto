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

root_folder = os.getcwd() + "\data"

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




# Encoder / Decoder LSTM model
def end_dec_lstm(x_steps, y_steps, n_features):
    model = Sequential([
        LSTM(200, activation = "relu", input_shape = (x_steps, n_features)), 
        RepeatVector(y_steps),
        LSTM(100, activation = "relu", return_sequences = True), 
        TimeDistributed(Dense(n_features))    
    ])
    return model

# Compiler
def base_compiler(model):
    return model.compile(loss=tf.losses.MeanSquaredError(),
                    optimizer=tf.optimizers.Adam(learning_rate = 0.0005),
                    metrics=[tf.metrics.MeanAbsoluteError()])



wide_close[assets] = wide_close[assets].replace([np.inf, -np.inf], np.nan)

def full_model(data_in, col, x_steps, y_steps, fun, compiler,
               save_loc, save = True, epochs = 50, patience = 2):
    
    n_features = len(col)
    # TODO: Find a way to keep track of timestamps
    
    # TODO: is there a faster way to do this?
    def split_sequences(sequences, n_steps_in, n_steps_out):
    	X, y = list(), list()
    	for i in range(len(sequences)):
    		# find the end of this pattern
    		end_ix = i + n_steps_in
    		out_end_ix = end_ix + n_steps_out
    		# check if we are beyond the dataset
    		if out_end_ix > len(sequences):
    			break
    		# gather input and output parts of the pattern
    		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
    		X.append(seq_x)
    		y.append(seq_y)
    	return np.array(X), np.array(y)
    
    t0 = time()
    all_x, all_y = split_sequences(np.array(data_in[col]), x_steps, y_steps)
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
    filt_x, filt_y, both = remove_nas_3d(all_x.reshape(all_x.shape[0], all_x.shape[1], n_features), 
                                   all_y.reshape(all_y.shape[0], all_y.shape[1], n_features))
    t1 = time()
    print("remove_nas_3d took " + str(round(t1 - t0)) + " seconds")
    
    # Saving timestamps
    time_filt = np.array(data_in["time"][(x_steps-1):(len(data_in)-y_steps)][both])
    
    # Train / Test Splits
    # TODO: Spit into folds for CV
    # TODO: Normalise again here
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
    del([all_x, all_y, filt_x, filt_y])    
    
    # Get Model, specified in fun
    model = fun(x_steps, y_steps, n_features)    
    
    # Compile model with compiler
    compiler(model)
    
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')
    
    model.summary()
    
    history = model.fit(train_x, train_y, epochs = epochs, 
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
    full_model(wide_close, "1", 60, 15, 1, simple_lstm, base_compiler)

# !mkdir "models"
# !mkdir "models\base_lstm_model"

base_lstm_model.save("models/base_lstm_model")

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




