import numpy as np
import math

def predict(X, y):
    """
    Predicts a matrix of predicted y values based on the inputted X values

    Args:
        X (np.array): matrix of input values
        Y (np.array): matrix of target values

    Returns:
        (np.array) The predicted y values based on a LSRL using the inputted X Values
    """

    X_mean = np.mean(X) # Mean of all of our inputs 
    y_mean = np.mean(y) # Mean of all of our targets

    # Sum of: inputs - mean of inputs * targets - mean of targets
    numerator = np.sum((X - X_mean) * (y - y_mean)) 

    # Sum of: inputs - mean of inputs (squared)
    denominator = np.sum((X - X_mean) ** 2)

    # Slope is just numerator / denominator (${})
    slope = numerator / denominator

    # Y-intercept is mean of targets - slope * mean of inputs
    intercept = y_mean - slope * X_mean

    # Finally, we will compute the predicted matrix which is our inputs * slope + intercept
    predicted_y = X * slope + intercept
    return predicted_y

def get_rmse(y, predicted_y):
    """
    Calculates the Root Mean Square Error of the LSRL that is being used

    Args:
        y (np.array): original target values for your data
        predicted_y (np.array): array of predicted target values based on the input feature in your data

    Returns:
        (float) The RMSE of the LSRL from the data's feature
    """

    return math.sqrt(1/len(y) * np.sum((y - predicted_y) ** 2))

def get_r_squared(y, predicted_y):
    """
    Calculates the R^2 Score of the LSRL that is being used

    Args:
        y (np.array): original target values for your data
        predicted_y (np.array): array of predicted target values based on the input features in your data

    Returns:
        (float) The R^2 Score of the LSRL from the data's feature

    """
    return 1 - (np.sum((y - predicted_y) ** 2) / np.sum((y - np.mean(y)) ** 2))