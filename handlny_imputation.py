import numpy as np
import datawig
import math
from geopy.geocoders import Nominatim
import pandas as pd


def imputate(data, target_column, num_epochs, logs_path):
    null_rows = data[target_column].isnull()
    df_train, df_test = datawig.utils.random_split(data)
    imputer = datawig.SimpleImputer(
        input_columns=data.columns,  # column(s) containing information about the column we want to impute
        output_column=target_column,  # the column we'd like to impute values for
        output_path=logs_path  # stores model data and metrics
    )
    imputer.fit(train_df=df_train, num_epochs=num_epochs, patience=num_epochs)
    imputed = imputer.predict(df_test)
    mse = np.mean((imputed[target_column + "_imputed"] - imputed[target_column]) ** 2) ** 0.5

    imputed.at[null_rows, target_column] = imputed[null_rows][target_column + "_imputed"]
    imputed.drop(target_column + "_imputed", axis=1)
    return imputed, mse


def clean_location_string(data):
    data = data.str.replace('\d+', '')
    data = data.str.replace('+.*+.∗', '')
    data = data.str.replace('\.', '')
    data = data.str.replace(' no.', '')
    return data


def get_location(countries):
    countries = clean_location_string(countries)
    geolocator = Nominatim(user_agent="specify_your_app_name_here")
    y = 0
    data = pd.DataFrame(index=range(len(countries)), columns=['latitude', 'longitude'])
    for country in countries:
        try:
            location = geolocator.geocode(country)
            data['latitude'][y] = location.latitude
            data['longitude'][y] = location.longitude
        except:
            data['latitude'][y] = None
            data['longitude'][y] = None
        y += 1
    return data['latitude'], data['longitude']


