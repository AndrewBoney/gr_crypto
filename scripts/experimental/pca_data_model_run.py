# PCA Data
# Using preprocessed data in R. Taking a PCA (200 cols) of 60 mins of returns
# This worked pretty well in R. Got ~ 75% up/down accuracy
# I want to see how well a neural network will work
import os
import pandas as pd
import numpy as np
import tensorflow as tf

root_folder = os.getcwd() + "\data\working"
pca_data = pd.read_csv(root_folder + "\pca_data.csv").drop("time", axis = 1)

means = pca_data.mean()
stds = pca_data.std()

pca_data_norm = (pca_data - pca_data.mean()) / pca_data.std()

y = np.array(pca_data_norm["Bitcoin"]).reshape(-1, 1)
x = np.array(pca_data_norm.drop("Bitcoin", axis = 1))

# Train / Test Splits
# TODO: Spit into folds for CV
n = len(y)
train_x = x[0:int(n*0.8),:]
train_y = y[0:int(n*0.8),:]

test_x = x[int(n*0.8):,:]
test_y = y[int(n*0.8):,:]

print("train_x is ", train_x.shape)
print("train_y is ", train_y.shape)
print("test_x is ", test_x.shape)
print("test_y is ", test_y.shape)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=100, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(units=50, activation='relu'),
    tf.keras.layers.Dropout(0.1),
    tf.keras.layers.Dense(units=10, activation='relu'),
    tf.keras.layers.Dense(units=1)
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

model_history = compile_and_fit(model, train_x, train_y, patience = 2)

preds = model.predict(test_x)

