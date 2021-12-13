import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats as stats


#default chart background
sns.set()

def plot_variable_pairs(df):


    #define numeric columns
    quant_vars = ['bathroom_count', 'bedroom_count', 'square_footage', 'tax_value'] 

    #determine correlation coefficients
    corr1, p1 = stats.pearsonr(df[quant_vars[1]], df[quant_vars[0]])
    corr2, p2 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[0]])
    corr3, p3 = stats.pearsonr(df[quant_vars[2]], df[quant_vars[1]])
    corr4, p4 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[0]])
    corr5, p5 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[1]])
    corr6, p6 = stats.pearsonr(df[quant_vars[3]], df[quant_vars[2]])

    #plot relationships between continuous variables
    sns.lmplot(x = quant_vars[0], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr1, 3)} | P-value: {p1} \n -----------------');
    sns.lmplot(x = quant_vars[2], y = quant_vars[0], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr2, 3)} | P-value: {p2} \n -----------------');
    sns.lmplot(x = quant_vars[2], y = quant_vars[1], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr3, 3)} | P-value: {p3} \n -----------------');
    sns.lmplot(x = quant_vars[0], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr4, 3)} | P-value: {p4} \n -----------------');
    sns.lmplot(x = quant_vars[1], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr5, 3)} | P-value: {p5} \n -----------------');
    sns.lmplot(x = quant_vars[2], y = quant_vars[3], data = df, line_kws = {'color': 'purple'})
    plt.title(f'R-value: {round(corr6, 3)} | P-value: {p6} \n -----------------');


def plot_categorical_and_continuous(df, x, y):

    sns.set()

    plt.figure(figsize = (8, 5))
    sns.boxplot(x = x, y = y, data = df, palette = 'inferno_r');
    plt.figure(figsize = (8, 5))
    sns.stripplot(x = x, y = y, data = df, palette = 'inferno_r');
    plt.figure(figsize = (8, 5))
    sns.violinplot(x = x, y = y, data = df, palette = 'inferno_r');