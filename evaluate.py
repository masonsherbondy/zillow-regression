import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mason_functions as mf




def plot_residuals(y, y_hat):
    #calculate residuals
    residuals = y - y_hat
    
    #create figure
    plt.figure(figsize = (8, 5))
    
    #plot residuals
    plt.scatter(y, residuals, color = 'slategray')
    
    #plot a dotted horizontal line at 0 to help visualize residual values
    plt.axhline(y = 0, ls = ':', color = 'firebrick')
    
    #clarity
    plt.title('Residuals From Predicted Values', pad = 7, size = 17, color = 'slategray')
    plt.xlabel('Actual Value', size = 15)
    plt.ylabel('Residual', size = 15, color = 'slategray')


def regression_errors(y, y_hat):
    #calculate residuals
    residuals = y - y_hat
    
    #residuals squared
    residuals_squared = residuals ** 2
    
    #sum of squared errors
    SSE = residuals_squared.sum()
    
    #explained sum of squares
    ESS = sum((y_hat - y.mean())**2)
    
    #total sum of squares
    TSS = ESS + SSE
    
    #mean of squared errors
    MSE = SSE / len(y)
    
    #root of mean of squared errors
    RMSE = MSE ** (1/2)
    
    #gimme gimme
    print('Model Metrics')
    print('=============')
    return pd.Series({
        'SSE': SSE,
        'ESS': ESS,
        'TSS': TSS,
        'MSE': MSE,
        'RMSE': RMSE
    })


def baseline_mean_errors(y):
    #determine baseline
    baseline = y.mean()
    
    #calculate baseline residuals
    baseline_residuals = y - y.mean()
    
    #sum of squared errors (baseline)
    SSE = (baseline_residuals ** 2).sum()
    
    #mean of squared errors (baseline)
    MSE = SSE / len(y)
    
    #root of mean of squared errors (baseline)
    RMSE = MSE ** (1/2)
    
    print('Baseline Metrics')
    print('----------------')
    return pd.Series({
        'SSE': SSE,
        'MSE': MSE, 
        'RMSE': RMSE
    })


def better_than_baseline(y, y_hat):
    #calculate residuals
    residuals = y - y_hat
    
    #residuals squared
    residuals_squared = residuals ** 2
    
    #sum of squared errors
    SSE = residuals_squared.sum()
    
    #explained sum of squares
    ESS = sum((y_hat - y.mean())**2)
    
    #total sum of squares
    TSS = ESS + SSE
    
    #mean of squared errors
    MSE = SSE / len(y)
    
    #root of mean of squared errors
    RMSE = MSE ** (1/2)
    
    #determine baseline
    baseline = y.mean()
    
    #calculate baseline residuals
    baseline_residuals = y - y.mean()
    
    #sum of squared errors (baseline)
    SSE_baseline = (baseline_residuals ** 2).sum()
    
    #mean of squared errors (baseline)
    MSE_baseline = SSE_baseline / len(y)
    
    #root of mean of squared errors (baseline)
    RMSE_baseline = MSE_baseline ** (1/2)
    
    #return bool True if mine model is better; return bool False if no
    return RMSE < RMSE_baseline