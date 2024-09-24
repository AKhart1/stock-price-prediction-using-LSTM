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

plt.rcParams['figure.figsize'] = (20,20)
figure, axes = plt.subplots(6)

for ax, col in zip(axes, df_scaled.columns):
    ax.plot(df_scaled[col])
    ax.set_title(col)
    ax.axes.xaxis.set_visible(False)

#plt.show()

def create_sequence(data, window_size):
    X = []
    Y = []
    for i in range(window_size, len(data)):
        X.append(data.iloc[i-window_size:i].values)
        Y.append(data.iloc[i].values)
    return np.array(X), np.array(Y)

window_size = 60
X,Y = create_sequence(df_scaled, window_size)

#print(X.shape, '\n', Y.shape)
#print(X)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
print(X_train.shape, X_test.shape)

## Building LSTM Model
model = keras.Sequential([
    #First layer
    keras.layers.LSTM(units = 50, return_sequence = False, input_shape = (X_train.shape[1], X_train.shape[2]))
])
