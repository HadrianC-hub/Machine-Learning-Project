"""Model training and persistence utilities."""

import os
import json
from datetime import datetime
from typing import Optional, Dict, Any, Union
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression


def train_logistic_regression(
    X_train: Union[pd.DataFrame, np.ndarray],
    y_train: Union[pd.Series, np.ndarray],
    model_name: str = 'logistic_regression',
    save_path: str = 'src/models/',
    **hyperparams
) -> LogisticRegression:
    """
    Train a Logistic Regression model.
    
    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training target
    model_name : str, default='logistic_regression'
        Name identifier for the model
    save_path : str, default='src/models/'
        Directory to save the model
    **hyperparams
        Additional hyperparameters for LogisticRegression
        (e.g., C, max_iter, penalty, solver)
        
    Returns:
    --------
    LogisticRegression
        Trained model
        
    Examples:
    ---------
    >>> model = train_logistic_regression(X_train, y_train)
    >>> model = train_logistic_regression(X_train, y_train, C=0.1, max_iter=1000)
    """
    # Convert to numpy arrays if needed
    if isinstance(X_train, pd.DataFrame):
        X_array = X_train.values
    else:
        X_array = X_train
    
    if isinstance(y_train, pd.Series):
        y_array = y_train.values
    else:
        y_array = y_train
    
    # Default hyperparameters
    default_params = {
        'random_state': 42,
        'max_iter': 1000,
        'solver': 'lbfgs'
    }
    
    # Merge with user-provided hyperparameters
    params = {**default_params, **hyperparams}
    
    # Initialize and train model
    model = LogisticRegression(**params)
    model.fit(X_array, y_array)
    
    return model


def save_model(
    model: Any,
    model_name: str,
    save_path: str = 'src/models/',
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, str]:
    """
    Save a trained model and its metadata.
    
    Parameters:
    -----------
    model : Any
        Trained model to save
    model_name : str
        Name identifier for the model
    save_path : str, default='src/models/'
        Directory to save the model
    metadata : dict, optional
        Additional metadata to save (hyperparameters, metrics, etc.)
        
    Returns:
    --------
    dict
        Dictionary with paths to saved model and metadata files
        
    Examples:
    ---------
    >>> paths = save_model(model, 'logistic_regression', metadata={'accuracy': 0.95})
    """
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Generate timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save model
    model_filename = f"{model_name}_{timestamp}.joblib"
    model_path = os.path.join(save_path, model_filename)
    joblib.dump(model, model_path)
    
    # Prepare metadata
    model_metadata = {
        'model_name': model_name,
        'model_path': model_path,
        'timestamp': timestamp,
        'training_date': datetime.now().isoformat(),
    }
    
    # Add model-specific attributes if available
    if hasattr(model, 'get_params'):
        model_metadata['hyperparameters'] = model.get_params()
    
    # Add custom metadata
    if metadata:
        model_metadata.update(metadata)
    
    # Save metadata
    metadata_filename = f"{model_name}_{timestamp}_metadata.json"
    metadata_path = os.path.join(save_path, metadata_filename)
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2, default=str)
    
    return {
        'model_path': model_path,
        'metadata_path': metadata_path
    }


def load_model(
    model_name: str,
    model_path: Optional[str] = None
) -> Any:
    """
    Load a saved model.
    
    Parameters:
    -----------
    model_name : str
        Name identifier for the model
    model_path : str, optional
        Full path to the model file. If None, searches in src/models/
        
    Returns:
    --------
    Any
        Loaded model
        
    Raises:
    -------
    FileNotFoundError
        If model file is not found
        
    Examples:
    ---------
    >>> model = load_model('logistic_regression', 'src/models/logistic_regression_20240101_120000.joblib')
    """
    if model_path is None:
        # Search for the most recent model with this name
        models_dir = 'src/models/'
        if not os.path.exists(models_dir):
            raise FileNotFoundError(f"Models directory not found: {models_dir}")
        
        # Find all models with this name
        matching_files = [
            f for f in os.listdir(models_dir)
            if f.startswith(model_name) and f.endswith('.joblib')
        ]
        
        if not matching_files:
            raise FileNotFoundError(
                f"No model found with name '{model_name}' in {models_dir}"
            )
        
        # Get the most recent one (by filename timestamp)
        matching_files.sort(reverse=True)
        model_path = os.path.join(models_dir, matching_files[0])
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = joblib.load(model_path)
    return model

