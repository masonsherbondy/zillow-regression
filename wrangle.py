import numpy as np
import pandas as pd
import mason_functions as mf
import os
import warnings
warnings.filterwarnings('ignore')

def acq_zillow_nadir():
        
    '''
    This function reads the data from a relational database into a dataframe and writes my initial dataframe to a .csv (in case the kernel gotta go)
    TL; DR: This function acquires the data.
    '''
    
    #define my sql query
    sql = '''
    SELECT bathroomcnt, bedroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt
    FROM predictions_2017
    LEFT JOIN properties_2017 USING(parcelid)
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


def clean_zillow_nadir():

    '''
    This function acquires my zillow data, drops any observations where the values are missing, and renames the columns so that they are easier to work with.
    TL; DR: This function cleans my data.
    '''

    #acquire data
    zillow_nadir = acq_zillow_nadir()

    #rename columns
    zillow_nadir = zillow_nadir.rename(columns = {'bathroomcnt': 'bathroom_count',
                  'bedroomcnt': 'bedroom_count',
                  'calculatedfinishedsquarefeet': 'square_footage',
                  'taxvaluedollarcnt': 'tax_value'
                  })

    #drop rows where values are missing (less than 0.2% of observations)
    zillow_nadir = zillow_nadir[zillow_nadir.square_footage.notnull()]
    zillow_nadir = zillow_nadir[zillow_nadir.tax_value.notnull()]
    
    #return cleaned data
    return zillow_nadir
    
    
def prep_zillow_nadir():

    '''
    This function acquires my zillow mvp data, cleans it, and returns my train, validate and test data sets for exploration and modeling.
    '''

    #clean the dataframe
    zillow_nadir = clean_zillow_nadir()

    #split the data with a function from my personal .py (using sklearn.model_selection)
    train, validate, test = mf.split_data(zillow_nadir)

    #return the data
    return train, validate, test
    