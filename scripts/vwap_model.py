# G-Research Crypto Kaggle Project

# Running tf model based on vwap.
# I want to try this a couple of different ways :
#   - Normalised (whole column)
#   - Normalised (by each "row" in the 3d array)
# I think row based will be better, as Target is Growth not absolute

# Setup 
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import tensorflow as tf
import multiprocessing

cores = multiprocessing.cpu_count() 
tf.config.threading.set_inter_op_parallelism_threads(cores-1)
tf.keras.backend.set_floatx('float64')

root_folder = os.getcwd() + "/data"

# Reading
wide_vwap = pd.read_csv(root_folder + "/working/wide_vwap.csv")
wide_high = pd.read_csv(root_folder + "/working/wide_vwap.csv")
wide_low = pd.read_csv(root_folder + "/working/wide_low.csv")
wide_target = pd.read_csv(root_folder + "/working/wide_target.csv")
asset_details = pd.read_csv(root_folder + "/asset_details.csv")

assets = list(asset_details["Asset_Name"])

# Preprocess DataFrame
## Getting high_rel and low_rel
high_rel = pd.DataFrame(columns=assets, index=range(len(wide_low)))
low_rel = pd.DataFrame(columns=assets, index=range(len(wide_low)))

[s + mystring for s in mylist]

for a in assets:    
    low_rel[a + "_low"] = wide_low[a] / wide_vwap[a]
    high_rel[a + "_high"] = wide_high[a] / wide_vwap[a]

## Converting Target df to just bitcoin, for merging on
btc = wide_target[["time", "Bitcoin"]]
btc = btc.rename(columns = {"Bitcoin": "Target"})

df = wide_vwap.merge(btc, how = "left", on = "time")

## Convert Inf to NA. Makes dropna() work and generally easier to work with
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Changes of "Target" above or below 0.025 I think are causing problems. 
# I'm going to manually set to NA for now, a more elegant solution may follow
df["Target"] = np.where(np.abs(df["Target"]) > 0.025,  np.nan, df["Target"])

## Checking that we don't have missing minutes
print(df.shape)
# print(sum(df["time"] == (df["time"].shift(1) + pd.Timedelta(minutes = 1))))


# Filtering
"""
Before creating 3d Array, I'm going to filter for rows after all coins exist.
This just makes the dataset smaller. 
NB - dogecoin only exists after ~mid 2019, so removing na rows removes all data before
this point. If you want more data you could impute Dogecoin, although this of course 
comes with its own problems. 
"""
# NB - This doesn't seem to be working properly. I think there's some noise 
# in Dogecoin, where a random row is = 0 
first = df[assets].apply(pd.Series.first_valid_index)

df_filt = df[df.index >= max(first)]

## Drop time (not part of modelling)
df_filt = df_filt.drop("time", axis = 1)

outliers = np.abs(df_filt["Target"]) > 0.025
print("Number of outliers is ", np.sum(outliers))

## Normalise Data
means = df_filt.mean()
stds = df_filt.std()

df_filt_norm = (df_filt - df_filt.mean()) / df_filt.std()

long = df_filt_norm.dropna().melt(var_name = "column", value_name = "value")

ax = sns.violinplot(x='column', y='value', data = long)
_ = ax.set_xticklabels(long.keys(), rotation=90)



# Convert to np.array
## Get col index of df Target column
pos = df_filt_norm.columns.get_loc("Target")

## Convert to Array
data_arr = np.array(df_filt_norm)

print("Target col is " + str(pos))

# Clear some space

# TODO: Add pos logic, for getting target
# TODO: More efficient???
def create_3d(arr, sequence_len):
    all_x = np.zeros((len(arr) - sequence_len, sequence_len, 14))
    
    for c in range(0, 14):
        for r in range(sequence_len, len(arr) - 1):
            all_x[r - sequence_len, :sequence_len, c] = arr[(r-sequence_len):r, c]
        print("column " + str(c))
    
    all_y = arr[sequence_len:, 14].reshape(-1, 1) 
    
    return all_x, all_y

all_x, all_y = create_3d(data_arr, 30)

# Remove NAs
## Checking for X (e.g. 3D)
### Boolean for NA rows
missing_x = ~np.isnan(all_x).any(axis=1)
missing_row_x = ~np.logical_not(missing_x).any(axis=1)

## Checking for Y
missing_y = ~np.isnan(all_y).any(axis=1)

## Combining
both = np.logical_and(missing_row_x, missing_y)

## Filtering arrays
filt_x = all_x[both, :, :]
filt_y = all_y[both, :]


# Train / Test Splits
# TODO: Spit into folds for CV
n = len(filt_y)
train_x = filt_x[0:int(n*0.8),:,:]
train_y = filt_y[0:int(n*0.8),:]

test_x = filt_x[int(n*0.8):, :, :]
test_y = filt_y[int(n*0.8):,:]

print("train_x is ", train_x.shape)
print("train_y is ", train_y.shape)
print("test_x is ", test_x.shape)
print("test_y is ", test_y.shape)

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences = True, dropout = 0.4),
    tf.keras.layers.LSTM(32), 
    tf.keras.layers.Dense(16, activation='relu'),
    # Shape => [batch, time, features]
    tf.keras.layers.Dense(units=1)
])

mult_dense_mod = tf.keras.Sequential([
    # Shape: (time, features) => (time*features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=1000, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=500, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1]),
])

def compile_and_fit(model, x, y, patience=5, epochs = 10):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(x = x, y = y, epochs=epochs,
                      validation_split=0.2, callbacks=[early_stopping])
    return history

lstm_history = compile_and_fit(lstm_model, train_x, train_y, patience = 2, epochs = 10)
# dense_history = compile_and_fit(mult_dense_mod, train_x, train_y, patience = 2, epochs = 10)

preds = lstm_model.predict(test_x)

plt.hist(preds)

preds_actual = (preds * stds["Target"]) + means["Target"]
y_actual = (test_y * stds["Target"]) + means["Target"]

plt.hist(y_actual)
plt.hist(preds_actual)

from sklearn.metrics import confusion_matrix

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

evaluate_regression(preds_actual, y_actual)
evaluate_up_down(preds_actual, y_actual)



