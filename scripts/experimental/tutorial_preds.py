import os
import tensorflow as tf
import pandas as pd 
import numpy as np
import random

# Replicating Target
import os
import pandas as pd
import numpy as np
import matplotlib as plt

root_folder = os.getcwd() + "\data"

train = pd.read_csv(root_folder + "/train.csv")
asset_details = pd.read_csv(root_folder + "/asset_details.csv")

def calculate_target(data: pd.DataFrame, details: pd.DataFrame, price_column: str):
    ids = list(details.Asset_ID)
    asset_names = list(details.Asset_Name)
    weights = np.array(list(details.Weight))

    all_timestamps = np.sort(data['timestamp'].unique())
    targets = pd.DataFrame(index=all_timestamps)

    for i, id in enumerate(ids):
        asset = data[data.Asset_ID == id].set_index(keys='timestamp')
        price = pd.Series(index=all_timestamps, data=asset[price_column])
        targets[asset_names[i]] = (
            price.shift(periods=-16) /
            price.shift(periods=-1)
        ) - 1
    
    targets['m'] = np.average(targets.fillna(0), axis=1, weights=weights)
    
    m = targets['m']

    num = targets.multiply(m.values, axis=0).rolling(3750).mean().values
    denom = m.multiply(m.values, axis=0).rolling(3750).mean().values
    beta = np.nan_to_num(num.T / denom, nan=0., posinf=0., neginf=0.)

    meas = (beta * m.values).T

    targets = targets - (beta * m.values).T
    targets.drop('m', axis=1, inplace=True)
    
    return targets

targets = calculate_target(train, asset_details, "Close")


base_lstm_model = tf.keras.models.load_model("models/base_lstm_model")
base_lstm_data = np.load("data/model_data/base_lstm.npz")

test_x = base_lstm_data["test_x"]
test_y = base_lstm_data["test_y"]

print(test_x.shape, test_y.shape)

def make_preds(model, x, y):
    # Preds WIP
    preds = model.predict(test_x)
    
    def evaluate_regression(pred, actual):
        errors = pred - actual
        mse = np.square(errors).mean()
        rmse = np.sqrt(mse)
        mae = np.abs(errors).mean()
        print('Mean Absolute Error: {:.4f}'.format(mae))
        print('Root Mean Square Error: {:.4f}'.format(rmse))
    
    evaluate_regression(preds, test_y.reshape(preds.shape[0], preds.shape[1]))
    
    def plot_pred(preds, test_x, test_y, seed):
        random.seed(seed)
        rand = random.randint(0, preds.shape[0])
        print("seed is " + str(rand))
        
        preds_compare = preds[rand, :].reshape(-1, 1)
        x_compare = test_x[rand, :].reshape(-1, 1)
        y_compare = test_y[rand, :].reshape(-1, 1)
        
        actuals_plt = np.concatenate([x_compare, 
                                  y_compare])
        
        empty = np.empty((x_compare.shape[0], x_compare.shape[1]))
        empty[:] = np.nan
        preds_plt = np.concatenate([empty, preds_compare])
        
        plt.plot(actuals_plt, label = "actuals")
        plt.plot(preds_plt, label = "preds")
        plt.legend()
        plt.show()
        
    for i in range(1, 5):
        plot_pred(preds, test_x, test_y, i)
        
    return preds