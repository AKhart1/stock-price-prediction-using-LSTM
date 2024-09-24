import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras 


df = pd.read_csv('NVidia_stock_history.csv')
print(df.shape)
print(df.head(), '\n')
#Detect if some values are missing
print(df.isna().sum())

#Data Preprocessing

df['Date'] = pd.to_datetime(df['Date'], utc=True)
df.set_index('Date', inplace=True)
print(df.info())
print(df.head())

df.sort_index(inplace=True)

scaler = MinMaxScaler()
scaler_values = scaler.fit_transform(df[df.columns])
print(scaler_values)
df_scaled = pd.DataFrame(scaler_values, columns=df.columns, index=df.index)
print(df_scaled.head())