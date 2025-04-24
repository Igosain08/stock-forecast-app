from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import numpy as np

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

def train_model_with_early_stopping(model, X_train, y_train, epochs=50, batch_size=32, validation_split=0.2):
    """Train model with early stopping to prevent overfitting"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stopping],
        verbose=0
    )
    
    return model, history

def evaluate_model(model, X_test, y_test, scaler=None):
    """Evaluate model and return metrics"""
    # Make predictions
    predictions = model.predict(X_test)
    
    # If scaler provided, inverse transform the predictions and y_test
    if scaler:
        # Create placeholder arrays
        pred_array = np.zeros((len(predictions), X_test.shape[2]))
        pred_array[:, 0] = predictions.flatten()
        y_array = np.zeros((len(y_test), X_test.shape[2]))
        y_array[:, 0] = y_test
        
        # Inverse transform
        predictions = scaler.inverse_transform(pred_array)[:, 0]
        y_test_inv = scaler.inverse_transform(y_array)[:, 0]
    else:
        y_test_inv = y_test
    
    # Calculate metrics
    mse = np.mean((predictions - y_test_inv) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - y_test_inv))
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'predictions': predictions,
        'y_test': y_test_inv
    }