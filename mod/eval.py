import numpy as np
from sklearn.metrics import confusion_matrix

def evaluate_regression(actual, pred):
    errors = pred - actual
    mse = np.square(errors).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(errors).mean()
    print('Mean Absolute Error: {:.4f}'.format(mae))
    print('Root Mean Square Error: {:.4f}'.format(rmse))
    
def evaluate_up_down(actual, pred):
    pred_up = pred > 0
    actual_up = actual > 0
    print(confusion_matrix(actual_up, pred_up))