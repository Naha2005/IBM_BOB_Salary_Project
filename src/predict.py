"""
predict.py
----------
Prediction helper — loads the saved model and exposes a simple
predict_salary() function for use by the CLI app or any other caller.
"""

import os
import joblib
import numpy as np


MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'salary_model.pkl')

# Module-level cache so the model is only loaded once per session
_model = None


def _get_model():
    """Load model from disk (cached after first call)."""
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                "Trained model not found. Please run `python src/train_model.py` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_salary(years_experience: float) -> float:
    """
    Predict salary for a given number of years of experience.

    Parameters
    ----------
    years_experience : float
        Number of years of professional experience (must be >= 0).

    Returns
    -------
    float
        Predicted annual salary in USD.

    Raises
    ------
    ValueError
        If years_experience is negative.
    """
    if years_experience < 0:
        raise ValueError("Years of experience cannot be negative.")

    model = _get_model()
    X = np.array([[years_experience]])
    prediction = model.predict(X)[0]
    return round(float(prediction), 2)
