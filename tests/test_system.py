"""
Unit tests for the CI/CD pipeline.
Ensures generation of model and a successful dummy inference.
"""

import os
import pandas as pd
import joblib


def test_model_exists():
    """Ensure that the trained model file is generated and exists."""
    model_path = os.path.join("models", "cricket_model.pkl")
    assert os.path.exists(model_path), "Model not found. Run train.py first."


def test_dummy_inference():
    """Test that the model accepts input shape and outputs probability."""
    model_path = os.path.join("models", "cricket_model.pkl")

    # Only test inference if model generation didn't fail
    if os.path.exists(model_path):
        model = joblib.load(model_path)

        dummy_data = pd.DataFrame([{
            "team1": "Australia",
            "team2": "South Africa",
            "venue": "Melbourne Cricket Ground",
            "toss_winner": "Australia"
        }])

        pred = model.predict(dummy_data)
        probs = model.predict_proba(dummy_data)

        assert len(pred) == 1
        assert len(probs[0]) == 2
