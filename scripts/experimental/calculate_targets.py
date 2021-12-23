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

    targets = targets - (beta * m.values).T
    targets.drop('m', axis=1, inplace=True)
    
    return targets

targets = calculate_target(train, asset_details, "Close")


# Testing
wide_close = pd.read_csv(root_folder + "/working/wide_close.csv")

# How close is more simple prediction, that only takes the asset return over 
# 15 mins?
simple = (wide_close["1"].shift(-16) / wide_close["1"].shift(-1)) - 1
actual = targets["Bitcoin"].reset_index(drop = True)

comparison = simple / actual
comparison.replace([np.inf, -np.inf], np.nan, inplace = True)
comparison_filt = comparison[~np.isnan(comparison)]

# This actually missed by quite a lot then. The only values that seem to match 
# are the first few thousand, which I guess is because of the rolling function
plt.hist(comparison_filt, bins = 100)
prop = round(comparison_filt, 5) != 1
print("Proportion that don't match are " + str(np.sum(prop) / len(comparison_filt)))

# This means that, for now, target will probably have to be taken from the 
# existing multiplier, until we have a version that 



