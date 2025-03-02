import os
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
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from keras.src.callbacks.model_checkpoint import ModelCheckpoint
from keras.src.callbacks.reduce_lr_on_plateau import ReduceLROnPlateau
from datetime import datetime
import yfinance as yF
import random
from scripts.preprocess import preprocess_data
from scripts.train import train_model
from scripts.predict import make_predictions

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
        
def main():
    # Preprocess the data
    data = preprocess_data('data/stock_prices.csv')
    
    # Train the model
    model = train_model(data)
    
    # Make predictions
    predictions = make_predictions(model, data)
    
    # Visualize the results
    # ...existing code to visualize predictions...

if __name__ == "__main__":
    main()
