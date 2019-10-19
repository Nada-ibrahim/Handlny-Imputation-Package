import pandas as pd
import numpy as np
import re

def Calculate_equation(dataframe, equation): 
    if '=' in equation:
        dataframe.eval(equation, inplace = True)
    if '=' not in equation:
        dataframe['calculated using equation '+equation] = dataframe.eval(equation)
        return dataframe

def clean_column_names(df):
    regex = re.compile('[^a-zA-Z]')
    for i, col in enumerate(df.columns):
        df = df.rename(columns = {df.columns[i]:regex.sub('_', df.columns[i])}) 
    return df.columns