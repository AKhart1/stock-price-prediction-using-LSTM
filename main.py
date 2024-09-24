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