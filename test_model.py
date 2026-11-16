
# Test Model - Customer Intelligence & Predictive Retention Platform

from pathlib import Path
import joblib

# Project model path
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "best_churn_model.joblib"


def test_model_file_exists():
    assert MODEL_PATH.exists(), "Trained model file not found!"


def test_model_loads_successfully():
    model = joblib.load(MODEL_PATH)
    assert model is not None


def test_model_has_prediction_method():
    model = joblib.load(MODEL_PATH)
    assert hasattr(model, "predict")



def test_model_generates_prediction():
    model = joblib.load(MODEL_PATH)

    import pandas as pd

    # Create one sample using the model's expected input columns
    sample = pd.DataFrame(
        [[0] * len(model.feature_names_in_)],
        columns=model.feature_names_in_
    )

    prediction = model.predict(sample)

    assert len(prediction) == 1
    assert prediction[0] in model.classes_
