import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np

from app.pipelines.data_pipeline import DataPipeline
from app.models.lstm_model import LSTMModel
from app.config import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT


app = FastAPI(title="FinSight AI API")


# Load model once
device = "cuda" if torch.cuda.is_available() else "cpu"

model = LSTMModel(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
).to(device)

# In portfolio project, load saved weights if available
MODEL_PATH = "models/lstm_model.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.eval()


class PredictionResponse(BaseModel):
    probability_up: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict():

    df = DataPipeline().run()

    # Use last WINDOW_SIZE rows
    from app.config import WINDOW_SIZE
    features = [
        "return",
        "volatility",
        "hl_range",
        "log_volume",
        "sentiment",
    ]

    latest = df[features].tail(WINDOW_SIZE).values
    X = torch.tensor(latest, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        logits = model(X.to(device))
        prob = torch.sigmoid(logits).item()

    return {"probability_up": prob}
