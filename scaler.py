def scale_data():

    #pull in split data
    train, test, validate = prep_zillow_nadir()

    #define numeric columns
    quant_vars = ['bathroom_count', 'bedroom_count', 'square_footage', 'tax_value']

    #no need to scale target variable
    quant_vars.remove('tax_value')

    #create object
    scaler = sklearn.preprocessing.RobustScaler()

    #fit object
    scaler.fit(train[quant_vars])

    #transform data
    train_scaled = scaler.transform(train[quant_vars])
    validate_scaled = scaler.transform(validate[quant_vars])
    test_scaled = scaler.transform(test[quant_vars])


    #return data
    return train_scaled, validate_scaled, test_scaled