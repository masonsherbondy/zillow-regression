# About This Repository
This repository is for the demonstration and display of my Zillow Regression project.

# About This Project

## Project Goals
The goals of this project are to construct a machine learning Regression model that predicts tax assessed values of single-family properties using their attributes and to determine key drivers of said values. I will deliver a report for the data science team to read through and replicate so that they understand what steps were taken as well as why and what the outcome was. I will also make recommendations on what does and/ or does not work for home value predictions.

## Project Description
We want to be able to predict the property tax assessed values of single-family properties that had a transaction during 2017. In this project, we will analyze attributes of single-family properties and develop a model to predict property tax assessed values. Deliverables are a github repository that includes the necessary files and instructions to reproduce my work as well as a final report, and a live presentation in which I present and walk through the final report for the data science team.

## Steps to Reproduce
Clone this repository and copy personal env.py into repo to run (personal valid credentials necessary). Libraries used are numpy, pandas, matplotlib, seaborn and sklearn.

## Initial Questions
1. Are properties concentrated in any one county?
2. Are houses priced higher on average in one county over another?
3. Why do some properties have a much higher value than others when they are located so close to each other?
4. Is having one bathroom worse than having two bedrooms?

## Data Dictionary
 | Feature           | Datatype                  | Description                                               |
 |:------------------|:--------------------------|:----------------------------------------------------------|
 | parcel_id         | index (object)            | Numeric ID unique to each parcel                          |
 | fips_id           | 4561 non-null: int64      | Federal Information Processing number (county)            |
 | zip_code          | 4561 non-null: int64      | Zip code (generalized location)                           |
 | bathroom_count    | 4561 non-null: float64    | Number of bathrooms on the property                       |
 | bedroom_count     | 4561 non-null: float64    | Number of bedrooms on the property                        |
 | square_footage    | 4561 non-null: float64    | Total square feet of the property                         |
 | tax_value         | 4561 non-null: float64    | Tax assessed home value                                   |


### Wrangle

* Write modules with functions that acquire, prepare and split the data. 
    * Import modules into notebook and test functions
    * Document individual reasons for data prep

### Explore

* explore through visualizations and statistical testing
* summarize findings

### Modeling

### Select evaluation metric
* I used the RMSE (root mean squared error) as my evaluation metric, because it is basically the average error

### Evaluate baseline
* I am using the mean property value as a baseline

### Develop 3 models
I have used:
* an ordinary least squares linear model
* a LassoLars linear model
* a generalized linear model

### Evaluate on train data set
* test multiple models on train data set to get a top performing model

### Evaluate on validate data set
* test best models on validate set

### Evaluate top model on test data set
* my top model was an OLS model that I decided to move forward with on the test set

### Report