import os
import json
import pickle
import torch
import torch.nn as nn
import numpy as np
from flask import Blueprint, request, jsonify, current_app
from src.text_preproc import clean_text, text_to_summary_features
from src.train_text_model import MLPClassifier  # Import your model class

# -----------------------------
# Blueprint for text-related endpoints
# -----------------------------
bp = Blueprint('text_route', __name__)

# -----------------------------
# Paths to saved models/vectorizers
# [TASK 15: Deployed model as functional web application, 10 pts]
# -----------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODEL_DIR = os.path.join(BASE_DIR, "models/text")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_FILES = {
    'adamw': os.path.join(MODEL_DIR, "text_model_adamw.pt"),
    'adam': os.path.join(MODEL_DIR, "text_model_adam.pt"),
    'sgd': os.path.join(MODEL_DIR, "text_model_sgd.pt")
}
LOG_FILE = os.path.join(BASE_DIR, "predictions_log.jsonl")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device for inference:", device)

# -----------------------------
# Logging predictions
# [TASK 1: Production-grade deployment - logging & error handling, 10 pts]
# -----------------------------
def log_prediction(text, prediction, probability, optimizer='adamw'):
    entry = {
        "text": text,
        "prediction": prediction,
        "probability": probability,
        "optimizer": optimizer
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")

# -----------------------------
# Load vectorizer
# -----------------------------
with open(VECTORIZER_FILE, "rb") as f:
    vectorizer = pickle.load(f)

# -----------------------------
# Load PyTorch models for each optimizer
# -----------------------------
torch_models = {}
input_dim = 5000  # Must match the TF-IDF max_features
for opt, path in MODEL_FILES.items():
    model = MLPClassifier(input_dim)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    torch_models[opt] = model
    print(f"Loaded PyTorch model ({opt}) from {path}")

# -----------------------------
# Prediction route
# [TASK 1: Production-grade deployment - logging & error handling]
# -----------------------------
@bp.route("/predict_text", methods=["POST"])
def predict_text():
    """
    Expects JSON:
        { "text": "Your message here", "optimizer": "adamw" }
    Returns:
        { "prediction": "spam" or "ham", "probability": float, "summary_features": {...} }
    """
    try:
        data = request.json or {}
        text = data.get("text", "")
        optimizer = data.get("optimizer", "adamw").lower()

        if not text:
            return jsonify({"error": "No text provided"}), 400
        if optimizer not in torch_models:
            return jsonify({"error": f"Unknown optimizer '{optimizer}'"}), 400

        # -----------------------------
        # Preprocess text
        # [TASK 8: Comprehensive preprocessing]
        # -----------------------------
        cleaned = clean_text(text)
        X = vectorizer.transform([cleaned]).toarray().astype(np.float32)
        X_tensor = torch.from_numpy(X).to(device)

        # -----------------------------
        # Predict using selected optimizer model
        # [TASK 5: Trained model inference, PyTorch]
        # -----------------------------
        model = torch_models[optimizer]
        with torch.no_grad():
            logits = model(X_tensor)
            prob = torch.sigmoid(logits).item()
            pred_label = 'spam' if prob >= 0.5 else 'ham'

        # -----------------------------
        # [TASK 9: Feature engineering / additional info]
        # -----------------------------
        features = text_to_summary_features(text)

        # -----------------------------
        # Log prediction
        # [TASK 1: Production-grade deployment]
        # -----------------------------
        log_prediction(text, pred_label, prob, optimizer)

        return jsonify({
            "prediction": pred_label,
            "probability": prob,
            "summary_features": features,
            "optimizer": optimizer
        })

    except Exception as e:
        # -----------------------------
        # Error handling (production-grade)
        # [TASK 1: Production-grade deployment]
        # -----------------------------
        current_app.logger.error(f"Prediction failed: {e}")
        return jsonify({"error": "Prediction failed. Check logs for details."}), 500
