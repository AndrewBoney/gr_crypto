# G-Research Crypto Kaggle Project

# Import data, tidy and create "wide" DataFrames

# How do I do headers in Spyder?

# Setup
import os
import pandas as pd
import numpy as np

# Read Data 
## Get Data 
root_folder = os.getcwd() + "/data"
train = pd.read_csv(root_folder + "/train.csv")
asset_details = pd.read_csv(root_folder + "/asset_details.csv")

## Adding time to dataset 
train["time"] = train["timestamp"].astype("datetime64[s]")

## Merging asset name on
train = train.merge(asset_details, how = "left", on = "Asset_ID")

## Getting unique assets 
assets = train["Asset_Name"].unique().tolist()

# Wide x Var
def wide_by_var(data, names_var, values_var):
    # Getting Wide close
    wide = data.pivot(index = "time", columns = names_var, values = values_var)
    
    # Filling any missing rows
    wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), 
                                                  name = "time", freq = "1 min"), 
                        fill_value = np.nan)
    
    # Converting index to column
    wide = wide.reset_index()
    
    return wide

wide_close = wide_by_var(train, "Asset_ID", "Close")
wide_vwap = wide_by_var(train, "Asset_ID", "VWAP")
wide_target = wide_by_var(train, "Asset_ID", "Target")
wide_high = wide_by_var(train, "Asset_ID", "High")
wide_low = wide_by_var(train, "Asset_ID", "Low")

# Save as csv
wide_close.to_csv(root_folder + "/working/wide_close.csv", index = False)
wide_vwap.to_csv(root_folder + "/working/wide_vwap.csv", index = False)
wide_target.to_csv(root_folder + "/working/wide_target.csv", index = False)
wide_high.to_csv(root_folder + "/working/wide_high.csv", index = False)
wide_low.to_csv(root_folder + "/working/wide_low.csv", index = False)

