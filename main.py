import os
import blpapi
import numpy as np
import pandas as pd
import tensorflow as tf 
from turtle import color
from colorama import init
from matplotlib import scale
import matplotlib.pyplot as plt
from keras.api import callbacks
from codecs import ignore_errors
from subprocess import check_output 
from sys import builtin_module_names
from keras.api.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import PredictionErrorDisplay
from keras.src.callbacks.callback import Callback
from keras.src.saving.saving_api import load_model
from sklearn.model_selection import train_test_split
from keras.src.callbacks.model_checkpoint import ModelCheckpoint
from keras.src.callbacks.reduce_lr_on_plateau import ReduceLROnPlateau
from datetime import datetime
import yfinance as yF
import random

class Colors:
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    END_RESET = '\033[0m'

class CustomConsoleOutput(Callback):
    def __init__(self):
        super().__init__()
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        train_mae = logs.get('mae', logs.get('MeanAbsoluteError', 0))
        train_rmse = logs.get('rmse', logs.get('RootMeanSquaredError', 0))
        val_mae = logs.get('val_mae', logs.get('val_MeanAbsoluteError', 0))
        val_rmse = logs.get('val_rmse', logs.get('val_RootMeanSquaredError', 0))
        loss = logs.get('loss', logs.get('loss', 0))
        
        lr = self.model.optimizer.lr.numpy() if hasattr(self.model.optimizer, 'lr') else 0

        print(f"\r{Colors.OKBLUE}Epoch {epoch + 1:02d}{Colors.END_RESET} | "
              f"Train MAE: {Colors.OKGREEN}{train_mae:.4f}{Colors.END_RESET} | "
              f"Train RMSE: {Colors.OKGREEN}{train_rmse:.4f}{Colors.END_RESET} | "
              f"Val MAE: {Colors.OKGREEN}{val_mae:.4f}{Colors.END_RESET} | "
              f"Val RMSE: {Colors.OKGREEN}{val_rmse:.4f}{Colors.END_RESET} | "
              f"Loss: {Colors.OKGREEN}{loss:.4f}{Colors.END_RESET} | "
              f"LR: {Colors.OKGREEN}{lr:.1e}{Colors.END_RESET}\n")


# # LSTM model

# # Data Preprocessing
df = pd.read_csv('NVidia_stock_history.csv')
df['Date'] = pd.to_datetime(df['Date'], utc=True)
df.set_index('Date', inplace=True)

latest_date = df.index.max()
print(f'\t\t\nLatest date in datset: {latest_date}\t\t\n')

start_date = (latest_date + pd.Timedelta(days=1)).date()
today = datetime.now().date()

# Updating dataSet with Yahoo API
if start_date <= today:
    print(f"Fetching data from {start_date} to {today}")
    
    ticker = yF.Ticker('NVDA')
    new_data = ticker.history(start=start_date, end=today)
    
    if not new_data.empty:
        
        new_data.reset_index(inplace=True)
        new_data['Date']= pd.to_datetime(new_data['Date'], utc= True)
        new_data.set_index('Date', inplace=True)
        
        new_data = new_data.rename(columns={
            'Open': 'Open',
            'High': 'High',
            'Low': 'Low',
            'Close': 'Close',
            'Volume': 'Volume',
            'Dividends': 'Dividends',
            'Stock Splits': 'Stock Splits'
        })
        
        new_data = new_data[['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']]
                
        df = pd.concat([df, new_data])
        df = df[~df.index.duplicated(keep='first')]
        
        df.to_csv('NVidia_stock_history.csv')
        print("\nDataset updated with a new data.\n")
    else:
        print("\nNo new data available\n")
else:
    print("\nDataset is up-to-date\n")

print(df.info(),'\n')
# Check for NaN's values
print(df.isna().sum())
# print('Dataset \t\t\n[Columns]\n', df.columns)
# print('Dataset \t\t\n[Head]\n', df.head())

df.sort_index(inplace=True)
scaler = MinMaxScaler()
scaler_values = scaler.fit_transform(df[df.columns])
df_scaled = pd.DataFrame(scaler_values, columns=df.columns, index=df.index)

print('\nNormalized dataset\n')
print(df_scaled.head(),'\n')
# print(df_scaled.describe(),'\n')

plt.rcParams['figure.figsize'] = (20,20)
figure, axes = plt.subplots(6)
for ax, col in zip(axes, df_scaled.columns):
    ax.plot(df_scaled[col])
    ax.set_title(col)
    ax.axes.xaxis.set_visible(False)

def create_sequence(data, window_size):
    X = []
    Y = []
    for i in range(window_size, len(data)):
        X.append(data.iloc[i-window_size:i].values)
        Y.append(data.iloc[i].values)
    return np.array(X), np.array(Y)

window_size = 150
X,Y = create_sequence(df_scaled, window_size)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state = 42)
print(X_train.shape, X_test.shape)

print(X_train.shape[1], X_train.shape[2])
print('\nLSTM model:\n')

print("Check for Nan Values in Input data:")
print("Any Nan's in training data: ", np.isnan(X_train).sum())
print("Any Nan's in target data: ", np.isnan(Y_train).sum())
# print(df_scaled.isna().sum())

keras = tf.keras
model_path = 'model_checkpoint.keras'   

if os.path.exists(model_path):
    print("Loaded model from checkpoint")
    
    model = load_model(model_path)
else:
    print("\nNo checkpoint found, training a new model.\n")
    model = keras.Sequential([
        #LSTM layers
        keras.Input(shape=(X_train.shape[1], X_train.shape[2])),
        keras.layers.LSTM(units=70, return_sequences=True),
        keras.layers.Dropout(0.3),
    
        keras.layers.LSTM(units=70, return_sequences=True),
        keras.layers.Dropout(0.3),    
        
        keras.layers.LSTM(units=70, return_sequences=False),
        keras.layers.Dropout(0.3),
    
        keras.layers.Dense(Y_train.shape[1])
    ])
    
    model.summary()
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['RootMeanSquaredError','MeanAbsoluteError','MeanAbsolutePercentageError'])
    
early_stop = EarlyStopping(
                        monitor='val_loss',
                        patience=15,
                        restore_best_weights=True)    

checkpoint_callback = ModelCheckpoint(
                        filepath = model_path,
                        save_best_only = True,
                        monitor = 'val_loss')

scheduler = ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.2,
                        patience=5,
                        verbose=1,
                        min_lr=1e-5,
                        mode='auto'
)

lstm_model = model.fit(
                        X_train, Y_train,
                        validation_split= 0.2,
                        epochs=20,
                        batch_size=32,
                        callbacks=[early_stop, checkpoint_callback, scheduler, CustomConsoleOutput()])

model.save(model_path)
print("Model saved.")

# Function to select a random year for predictions
def get_random_year(df):
    years = df.index.year.unique()
    return random.choice(years)

# # Make predictions
random_year = get_random_year(df_scaled)
print(f"Making predictions for the year: {random_year}")

# Filter test data for the selected year
X_test_year = X_test[df_scaled.index.year == random_year]
Y_test_year = Y_test[df_scaled.index.year == random_year]

predictions = model.predict(X_test_year)

# Rescale predictions and test values to the original values
predictions = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(Y_test_year)

# Convert to actual values 
predictions_df = pd.DataFrame(
    data= predictions[:,3],
    index= df_scaled[df_scaled.index.year == random_year].index,
    columns= ['Predicted']
)

comparison_df = pd.DataFrame({
    'Date': predictions_df.index,
    'Predicted[CL]': predictions_df['Predicted'],
    'Actual': y_test_rescaled[:,0],
    'Difference [%]': (abs(predictions_df['Predicted'] - y_test_rescaled[:,0]) / y_test_rescaled[:,0]) * 100

}).set_index('Date')
print(comparison_df.head())

# # Plot training and validation loss
plt.plot(lstm_model.history['loss'], label='Training loss')
plt.plot(lstm_model.history['val_loss'], label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()  
    
# # Plot the results
plt.figure(figsize= (14, 7))

for i, column in enumerate(df_scaled.columns):
    plt.subplot(3,3, i+1)
    plt.plot(y_test_rescaled[:,i], color='blue', label=f'Actual {column}')
    plt.plot(predictions[:, i], color='red', label=f'Predicted {column}')
    plt.title(f'{column} Prediction')
    plt.xlabel('Time')
    plt.ylabel(f'{column} price')
    plt.legend()
    
plt.tight_layout()
plt.show()
