import streamlit as st
import torch
import numpy as np

from app.pipelines.data_pipeline import DataPipeline
from app.models.lstm_model import LSTMModel
from app.config import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT, WINDOW_SIZE

st.set_page_config(page_title="FinSight AI", layout="wide")

st.title("📈 FinSight AI – NIFTY 50 Intelligence Platform")

st.markdown("Sentiment-Aware LSTM Prediction for Next-Day NIFTY 50 Direction")

device = "cpu"

# Load model
model = LSTMModel(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)

MODEL_PATH = "models/lstm_model.pth"

if torch.cuda.is_available():
    device = "cuda"

model.to(device)

if st.button("Predict Next-Day Direction"):

    with st.spinner("Running model inference..."):

        df = DataPipeline().run()

        features = [
            "return",
            "volatility",
            "hl_range",
            "log_volume",
            "sentiment",
        ]

        latest = df[features].tail(WINDOW_SIZE).values
        X = torch.tensor(latest, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(X)
            prob = torch.sigmoid(logits).item()

        st.metric("Probability of Up Move", f"{prob:.2%}")

        if prob > 0.55:
            st.success("Bullish Signal")
        elif prob < 0.45:
            st.error("Bearish Signal")
        else:
            st.warning("Neutral Signal")
