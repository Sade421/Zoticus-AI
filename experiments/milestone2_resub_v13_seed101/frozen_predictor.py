"""Frozen Sentient AI specialty predictor (Milestone 2 resubmission)."""
from pathlib import Path
import joblib

_MODEL = joblib.load(Path(__file__).with_name("frozen_model.joblib"))
LABELS = list(_MODEL.classes_)


def predict_one(text):
    """Return one supported specialty label for pre-decision text."""
    return str(_MODEL.predict([text])[0])


def predict_proba_one(text):
    return dict(zip(LABELS, map(float, _MODEL.predict_proba([text])[0])))
