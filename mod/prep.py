import pandas as pd
import numpy as np
from time import time

"""
class full_model:
    def __init__(self, data_in):
        self.data_in = data_in
"""

"""
log_return:
    Calculate log return on a pandas series.
"""
def log_return(series, periods=1):
    return np.log(series).diff(periods=periods)


"""
log_return_np:
    Calculate log return on a 1d np array.
"""
def log_return_np(array, periods=1):
    # TODO: Check shape
    log = np.log(array)
    diff = np.diff(log)
    return diff


def preprocess(data_in, col_in, col_out, time_col, x_steps, y_steps):    
    # Get n features
    if isinstance(col_in, str):
        n_features_x = 1
    elif isinstance(col_in, list):
        n_features_x = len(col_in)
    else:
        raise TypeError("""
                        col_in should be a string indicating a column in 
                        data_in, or a list of column names
                        """)
                        
    if isinstance(col_out, str):
        n_features_y = 1
    elif isinstance(col_out, list):
        n_features_y = len(col_out)
    else:
        raise TypeError("""
                        col_out should be a string indicating a column in 
                        data_in, or a list of column names
                        """)                        
    
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
    
    n = len(filt_y)
    
    # Saving timestamps
    time_filt = np.array(data_in[time_col][max(ix_in):(len(data_in)-max(ix_out))][both])
    
    time_out = {"train":time_filt[0:int(n*0.8)],
                "test":time_filt[int(n*0.8):]}
    
    # Train / Test / Val Splits
    
    
    train_x = filt_x[0:int(n*0.8),:,:]
    train_y = filt_y[0:int(n*0.8),:,:]
    
    test_x = filt_x[int(n*0.8):, :,:]
    test_y = filt_y[int(n*0.8):,:,:]
    
    print("train_x is ", train_x.shape)
    print("train_y is ", train_y.shape)
    print("test_x is ", test_x.shape)
    print("test_y is ", test_y.shape)
    
    return train_x, test_x, train_y, test_y, time_out
