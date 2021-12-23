# -*- coding: utf-8 -*-
"""
Created on Sun Dec 19 15:17:04 2021

@author: Andre
"""

print(os.getcwd())

# Target is wide. Doing some more investigating here. 
print(df_filt["Target"].max())
print(df_filt["Target"].min())
plt.hist(df_filt["Target"], bins = 50)
plt.hist(df_filt_norm["Target"], bins = 50)

check = (df_filt_norm["Target"] * stds["Target"]) + means["Target"]

plt.hist(check, bins = 50)

check = df_filt.sort_values("Target")
outliers = np.abs(df_filt["Target"]) > 0.025
print("Number of outliers is ", np.sum(outliers))
