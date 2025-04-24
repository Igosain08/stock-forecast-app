import numpy as np
from sklearn.metrics import mean_squared_error
from model_utils import create_lstm_model, create_gru_model, create_hybrid_model, train_model_with_early_stopping

def create_ensemble_model(X_train, y_train, X_test, y_test, input_shape, n_models=3):
    """
    Create an ensemble of models and train them on the given data
    
    Parameters:
    -----------
    X_train, y_train: Training data
    X_test, y_test: Testing data
    input_shape: Shape of input for models
    n_models: Number of models in ensemble
    
    Returns:
    --------
    Dictionary with trained models and ensemble predictions
    """
    models = []
    predictions = []
    
    # Create different model types
    model_creators = [
        create_lstm_model,
        create_gru_model,
        create_hybrid_model
    ]
    
    # Train each model type and store predictions
    for i in range(min(n_models, len(model_creators))):
        # Create the model
        model = model_creators[i](input_shape)
        
        # Train the model
        model, history = train_model_with_early_stopping(model, X_train, y_train)
        
        # Get predictions
        pred = model.predict(X_test)
        predictions.append(pred)
        models.append(model)
    
    # Create ensemble prediction (average of all model predictions)
    ensemble_pred = np.mean(predictions, axis=0)
    
    # Calculate ensemble error
    ensemble_mse = mean_squared_error(y_test, ensemble_pred)
    
    return {
        'models': models,
        'individual_predictions': predictions,
        'ensemble_prediction': ensemble_pred,
        'ensemble_mse': ensemble_mse
    }

def predict_next_with_ensemble(ensemble, X_recent):
    """
    Make prediction with ensemble for next time step
    
    Parameters:
    -----------
    ensemble: Dictionary with trained models
    X_recent: Most recent data point for prediction
    
    Returns:
    --------
    Ensemble prediction for next time step
    """
    # Get predictions from each model
    predictions = []
    for model in ensemble['models']:
        pred = model.predict(X_recent)
        predictions.append(pred)
    
    # Average predictions
    ensemble_pred = np.mean(predictions, axis=0)
    
    return ensemble_pred[0][0]