import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def windowed_dataframe_to_date_X_y(windowed_dataframe):
    # Convert windowed dataframe to numpy arrays for X and y
    df_as_np = windowed_dataframe.to_numpy()
    
    # Get dates as first column
    dates = df_as_np[:, 0]
    
    # Get middle columns for features (X) - all columns except first (date) and last (target)
    middle_matrix = df_as_np[:, 1:-1]
    
    # Reshape X to be 3D: [samples, time steps, features]
    X = middle_matrix.reshape(len(middle_matrix), middle_matrix.shape[1], 1)
    
    # Get last column as target (y)
    y = df_as_np[:, -1]
    
    return dates, X.astype(np.float32), y.astype(np.float32)

def df_to_windowed_df(dataframe, first_date, last_date=None, n=3):
    # Ensure timezone-aware timestamps
    first_date = pd.Timestamp(first_date).tz_localize('America/New_York') if first_date.tzinfo is None else first_date
    if last_date:
        last_date = pd.Timestamp(last_date).tz_localize('America/New_York') if last_date.tzinfo is None else last_date
    
    target_date = first_date
    dates = []
    X = []
    Y = []
    
    last_time = False
    
    while True:
        # Create window of size n ending on target_date
        df_subset = dataframe.loc[:target_date].tail(n+1)
        
        # Break if we don't have enough data for a complete window
        if len(df_subset) < n+1:
            break
        
        # Extract X and y values
        values = df_subset['Close'].values
        x, y = values[:-1], values[-1]
        
        # Store the results
        dates.append(target_date)
        X.append(x)
        Y.append(y)
        
        # Find next date
        next_date = dataframe.loc[target_date:].index[0] + pd.Timedelta(days=1)
        
        # Get the next business day (index date)
        next_row = dataframe.loc[next_date:].head(1)
        
        # If we've reached the last valid date or specified last_date, break
        if next_row.empty or (last_date and target_date >= last_date):
            if not last_time:
                # Set flag for one more iteration
                last_time = True
                next_row = dataframe.loc[target_date:].head(1)
            else:
                break
        
        # Update target_date for the next iteration
        target_date = next_row.index[0]
        
        # If we've specified a last_date and we're already past it, break
        if last_date and target_date > last_date:
            break
    
    # Build the windowed DataFrame
    df_dict = {'date': dates}
    
    # Add X columns
    for i in range(n):
        df_dict[f'x{i}'] = [x[i] for x in X]
    
    # Add y column
    df_dict['y'] = Y
    
    return pd.DataFrame(df_dict)

def create_windowed_data(dataframe, window_size=60):
    """Create windowed data for training LSTM models"""
    # Convert to numpy array
    data = dataframe.values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Create X and y sequences
    X, y = [], []
    for i in range(len(scaled_data) - window_size):
        X.append(scaled_data[i:i+window_size])
        y.append(scaled_data[i+window_size])
    
    # Create X_recent for prediction
    X_recent = scaled_data[-window_size:]
    
    # Convert to numpy arrays
    X, y = np.array(X), np.array(y)
    
    # Get dates corresponding to y values
    dates = dataframe.index[window_size:]
    
    return X, y, dates, X_recent, scaler

def create_lstm_model(input_shape, units=50, dropout_rate=0.2):
    """Create an enhanced LSTM model with dropout for regularization"""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def create_gru_model(input_shape, units=50, dropout_rate=0.2):
    """Create a GRU model which can be faster than LSTM but still effective"""
    model = Sequential([
        GRU(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        GRU(units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def create_hybrid_model(input_shape, units=50, dropout_rate=0.2):
    """Create a hybrid model combining LSTM and GRU"""
    model = Sequential([
        LSTM(units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        GRU(units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    
    model.compile(optimizer='adam', loss='mse')
    return model

def run_lstm_forecast(Close, model_type='lstm'):
    """Run LSTM forecast with selected model type"""
    # Create a dataframe from the Close price series
    df = pd.DataFrame(Close)
    
    # Create windowed data for LSTM
    X, y, dates, X_recent, scaler = create_windowed_data(df)
    
    # Split into training and testing sets (80/20)
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Select the model type
    input_shape = (X.shape[1], X.shape[2])
    if model_type == 'lstm':
        model = create_lstm_model(input_shape)
    elif model_type == 'gru':
        model = create_gru_model(input_shape)
    elif model_type == 'hybrid':
        model = create_hybrid_model(input_shape)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Early stopping to prevent overfitting
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    # Train the model with early stopping
    model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping],
        verbose=0
    )
    
    # Make predictions on test data
    y_pred = model.predict(X_test)
    
    # Inverse transform the scaled values
    y_test_inv = scaler.inverse_transform(np.concatenate([np.zeros((len(y_test), X.shape[2]-1)), y_test], axis=1))[:, -1]
    y_pred_inv = scaler.inverse_transform(np.concatenate([np.zeros((len(y_pred), X.shape[2]-1)), y_pred], axis=1))[:, -1]
    
    # Calculate evaluation metrics
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    
    # Predict the next day's price
    next_pred = model.predict(X_recent.reshape(1, X_recent.shape[0], X_recent.shape[1]))
    next_pred_inv = scaler.inverse_transform(np.concatenate([np.zeros((1, X.shape[2]-1)), next_pred], axis=1))[0, -1]
    
    return next_pred_inv, mse, r2, mae
