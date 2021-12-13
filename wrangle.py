import numpy as np
import pandas as pd
import mason_functions as mf
import os
import warnings
warnings.filterwarnings('ignore')

def acq_zillow_nadir():
        
    '''
    This function reads the data from a relational database into a dataframe and writes my initial dataframe to a .csv (in case the kernel gotta go)
    TL; DR: This function acquires my data.
    '''
    
    #define my sql query
    sql = '''
    SELECT parcelid, fips, regionidzip, bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt
    FROM predictions_2017
    LEFT JOIN properties_2017 USING(parcelid)
    JOIN propertylandusetype USING(propertylandusetypeid)
    WHERE propertylandusetypeid = 261 or propertylandusetypeid = 279
    '''

    #define my url using a url-retrieval function from my personal .py
    url = mf.get_db_url('zillow')

    #set up an if-conditional to see if there is a .csv readily available
    if os.path.isfile('zillow_nadir.csv'):
        
        #if there is, render a workable dataframe from the .csv
        df = pd.read_csv('zillow_nadir.csv', index_col = 0)
        
    #if not, access the relational database, and then write data to a .csv for later ease of access
    else:
        df = pd.read_sql(sql, url)
        df.to_csv('zillow_nadir.csv')
    
    #return the dataframe
    return df


def remove_outliers(df, k, col_list):
    ''' 
    Removes outliers from a list of columns in a dataframe and returns the dataframe.
    '''
    
    for col in col_list:

        q1, q3 = df[col].quantile([.25, .75])  # get quartiles
        
        iqr = q3 - q1   # calculate interquartile range
        
        upper_bound = q3 + k * iqr   # get upper bound
        lower_bound = q1 - k * iqr   # get lower bound

        # return dataframe without outliers
        
        df = df[(df[col] > lower_bound) & (df[col] < upper_bound)]
        
    return df


def clean_zillow_nadir():

    '''
    This function acquires my zillow data, renames the columns so that they are easier to work with, drops any observations where the values are missing and drops outliers from the dataframe. 
    TL; DR: This function cleans my data.
    '''

    #acquire data
    zillow_nadir = acq_zillow_nadir()

    #rename columns
    zillow_nadir = zillow_nadir.rename(columns = {
        'parcelid': 'parcel_id',
        'fips': 'fips_id',
        'zipregionid': 'zip_code',
        'bathroomcnt': 'bathroom_count',
        'bedroomcnt': 'bedroom_count',
        'calculatedfinishedsquarefeet': 'square_footage',
        'taxvaluedollarcnt': 'tax_value'
        })

    #set index to unique parcel ID and reduce noise on region identifiers
    zillow_nadir = zillow_nadir.set_index('parcel_id')
    zillow_nadir.fips_id = zillow_nadir.fips_id.astype(int)
    zillow_nadir['zip_code'] = zillow_nadir['zip_code'].astype(int)
    
    
    #drop rows where values are missing (less than 0.2% of observations)
    zillow_nadir = zillow_nadir[zillow_nadir.square_footage.notnull()]
    zillow_nadir = zillow_nadir[zillow_nadir.zip_code.notnull()]
    zillow_nadir = zillow_nadir[zillow_nadir.tax_value.notnull()]
    
    #define numeric columns
    quant_vars = ['bathroom_count', 'bedroom_count', 'square_footage', 'tax_value']

    #remove outliers
    zillow_nadir = remove_outliers(zillow_nadir, 1.5, quant_vars)

    #return cleaned data
    return zillow_nadir
    
    
def prep_zillow_nadir():

    '''
    This function acquires my zillow mvp data, cleans it, and returns my train, validate and test data sets for exploration.
    '''

    #clean the dataframe
    zillow_nadir = clean_zillow_nadir()

    #split the data with a function from my personal .py (using sklearn.model_selection)
    train, validate, test = mf.split_data(zillow_nadir)

    #return the data
    return train, validate, test
    
