import numpy as np

def return_pred(test_x, test_y, model):
    
    last_x = test_x[:, test_x.shape[1]-1, :].reshape(-1)
    
    # If tf then assume "flatten" layer has been used
    if type(model).__name__ == "Sequential":
        pred_pre = model.predict(test_x).reshape(-1)
    else:
        pred_pre = model.predict(test_x.reshape(-1, test_x.shape[1])).reshape(-1)
    true_pre = test_y.reshape(-1)
    
    pred = np.log(pred_pre) - np.log(last_x)
    true = np.log(true_pre) - np.log(last_x)
    
    rm = ~np.isinf(pred) & ~np.isinf(true)
    
    pred = pred[rm]
    true = true[rm]

    return true, pred
