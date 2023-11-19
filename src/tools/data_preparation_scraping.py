#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 15:02:53 2023

@author: charlottepapelard
"""



import pandas as pd

# Load the data
data = pd.read_excel("raw_data.xlsx")

# Remove duplicates in the original DataFrame
data.drop_duplicates(inplace=True)

# Drop rows with missing values
data.dropna(inplace=True)

# Display data types of each column
data_types = data.dtypes
print(data_types)





