import streamlit as st
import yfinance as yf
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pandas as pd
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


from tensorflow.keras.models import load_model

def windowed_df_to_date_X_y(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

def df_to_windowed_df(dataframe, first_date, last_date, n=3):
    # Ensure timezone-aware timestamps
    first_date = pd.Timestamp(first_date).tz_localize('America/New_York') if first_date.tzinfo is None else first_date
    last_date  = pd.Timestamp(last_date).tz_localize('America/New_York') if last_date.tzinfo is None else last_date

    target_date = first_date
    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n+1)

        if len(df_subset) != n+1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        # Get the next date (1 step forward)
        next_rows = dataframe.loc[target_date:].head(2)
        if len(next_rows) < 2:
            break  # Reached the end
        next_date = next_rows.index[1]

        if last_time:
            break

        target_date = next_date
        if target_date == last_date:
            last_time = True

    # Build the windowed DataFrame
    ret_df = pd.DataFrame({'Target Date': dates})
    X = np.array(X)

    for i in range(n):
        ret_df[f'Target-{n-i}'] = X[:, i]

    ret_df['Target'] = Y
    return ret_df

def train(Close):
    df= Close
    df=pd.DataFrame(Close)
    df=df.copy()
    df=df.reset_index()
    df['Date']
    df['Date'] = pd.to_datetime(df['Date'])
    df1 = df.set_index('Date')  # Required for .loc to work by datetime
    df1 = df1.tz_convert('America/New_York')  # Ensure timezone matches

# Call with timezone-aware timestamps
    windowed_df = df_to_windowed_df(
        df1,
        pd.Timestamp('2022-10-30', tz='America/New_York'),
        pd.Timestamp('2025-03-30', tz='America/New_York'),
        n=20
        )
    dates, X, y = windowed_df_to_date_X_y(windowed_df)
    q_80 = int(len(dates) * .8)
    q_90 = int(len(dates) * .9)

    dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]

    dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
    dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]
    
    model = Sequential([layers.Input((20, 1)),
                    layers.LSTM(64),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(32, activation='relu'),
                    layers.Dense(1)])

    model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.001),
              metrics=['mean_absolute_error'])

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100)
    train_predictions = model.predict(X_train).flatten()
    val_predictions = model.predict(X_val).flatten()
    test_predictions = model.predict(X_test).flatten()
    

    y_pred = model.predict(X_test).flatten()
    y_true = y_test.flatten()


    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    window_size = 20
    last_window = Close['Close'].values[-window_size:]  # shape: (20,)
    model_input = last_window.reshape(1, window_size, 1)
    next_prediction = model.predict(model_input, verbose=0).flatten()[0]


    return next_prediction,mse,r2,mae



def run_lstm_forecast(Close):
    # Do all LSTM model loading, preprocessing, predicting
    
    next_prediction,mse,r2,mae=train(Close)
    return next_prediction,mse,r2,mae
    



