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
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

cores = multiprocessing.cpu_count() 
tf.config.threading.set_inter_op_parallelism_threads(cores-1)
tf.keras.backend.set_floatx('float64')

root_folder = os.getcwd() + "/data"

# Reading
wide_vwap = pd.read_csv(root_folder + "/working/wide_vwap.csv")
wide_high = pd.read_csv(root_folder + "/working/wide_high.csv")
wide_low = pd.read_csv(root_folder + "/working/wide_low.csv")
wide_target = pd.read_csv(root_folder + "/working/wide_target.csv")
asset_details = pd.read_csv(root_folder + "/asset_details.csv")

# assets = list(asset_details["Asset_Name"])
assets = [str(i) for i in asset_details["Asset_ID"]]

# Preprocess DataFrame
## Getting high_rel and low_rel
high_rel = pd.DataFrame(columns=[s + "_high" for s in assets], index=range(len(wide_low)))
low_rel = pd.DataFrame(columns=[s + "_low" for s in assets], index=range(len(wide_low)))

for a in assets:    
    high_rel[a + "_high"] = np.log(wide_high[a]) - np.log(wide_vwap[a])
    low_rel[a + "_low"] = np.log(wide_low[a]) - np.log(wide_vwap[a])

# Adding time back to df
high_rel["time"] = wide_high["time"]
low_rel["time"] = wide_low["time"]

## Get vwap diff
# define function to compute log returns
def log_return(series, periods=1):
    return np.log(series).diff(periods=periods)

# Get minute by minute vwap returns for each asset
df = wide_vwap[assets].apply(log_return)
df["time"] = wide_vwap["time"]

## Converting Target df to just bitcoin, for merging on
btc_num = asset_details[asset_details["Asset_Name"] == "Bitcoin"]["Asset_ID"] \
          .to_string(index = False).strip()

btc = wide_target[["time", btc_num]]
btc = btc.rename(columns = {btc_num: "Target"})

df = df.merge(btc, how = "left", on = "time") \
    .merge(high_rel, how = "left", on = "time") \
    .merge(low_rel, how = "left", on = "time")

## Convert Inf to NA. Makes dropna() work and generally easier to work with
df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Changes of "Target" above or below 0.025 I think are causing problems. 
# I'm going to manually set to NA for now, a more elegant solution may follow
outliers = np.abs(df["Target"]) > 0.025
print("Number of outliers is ", np.sum(outliers))

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

df_filt = df_filt.reset_index(drop = True)

## Drop time (not part of modelling)
time = df_filt["time"]
df_filt = df_filt.drop("time", axis = 1)

## Normalise Data
means = df_filt.mean()
stds = df_filt.std()

df_filt_norm = (df_filt - df_filt.mean()) / df_filt.std()
df_filt_norm["time"] = time

# long = df_filt_norm.dropna().melt(var_name = "column", value_name = "value")
# ax = sns.boxplot(x='column', y='value', data = long)
# _ = ax.set_xticklabels(long.keys(), rotation=90)

# Highs and Lows have long heads / tails respectively. Not sure how 
# much of a problem this will be at this point...

# PCA 
# Running PCA on non bitcoin columns
# TODO: Tidy up this whole section
cols = [btc_num, btc_num + "_low", btc_num + "_high", "Target"]

temp = df_filt_norm.dropna()
time = pd.to_datetime(temp["time"])

no_na = temp.drop("time", axis = 1)
no_na = no_na.reset_index(drop = True)

bitcoin = no_na[cols]
other = no_na.drop(cols, axis = 1)

pca = PCA()
pca.fit(other)

## Scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=2)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()

out_sum = np.cumsum(pca.explained_variance_ratio_)  
print ("Cumulative Prop. Variance Explained: ", out_sum)

# I'm going to take 10 for now, although as always with PCA this could be up for debate
# I'd also like to look at other techniques, e.g. Kernel PCA.
other_pca_np = pca.transform(other)
other_pca = pd.DataFrame(other_pca_np, 
             columns=["pca_" + str(i) for i in list(range(1, other_pca_np.shape[1]+1))])

everything = pd.concat([other_pca[["pca_" + str(i) for i in list(range(1, 11))]], 
                        bitcoin], axis = 1).set_index(time)

everything = everything.reindex(pd.date_range(everything.index.min(), 
                                         everything.index.max(), 
                                         name = "time", freq = "1 min"))

x = np.array(everything.drop("Target", axis = 1))
y = np.array(everything["Target"]).reshape(-1, 1)

# Clear some space
del(outliers)
del(other_pca_np)
del(other)
del(other_pca)
del(no_na)
del(df)
del(df_filt)
# del(long)
del(everything)

del([wide_high, wide_low, wide_target, wide_vwap])


# Create a 3D input
def create_dataset (X, y, time_steps = 1):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        v = X[i:(i+time_steps), :]
        Xs.append(v)
        ys.append(y[i+time_steps])
    return np.array(Xs), np.array(ys)

all_x, all_y = create_dataset(x, y, 60)

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
    tf.keras.layers.LSTM(52, return_sequences = True, dropout = 0.4),
    tf.keras.layers.LSTM(104, dropout = 0.2), 
    tf.keras.layers.Dense(52, activation='relu'),
    tf.keras.layers.Dense(26, activation='relu'),
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

plt.hist(y_actual, bins = 50)
plt.hist(preds_actual, bins = 50)

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



