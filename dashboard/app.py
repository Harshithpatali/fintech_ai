import streamlit as st
import torch
import numpy as np
import yfinance as yf
import pandas as pd
import pywt

st.set_page_config(page_title="FinSight AI", layout="centered")

st.title("📈 FinSight AI – NIFTY 50 Direction Predictor")

st.markdown("LSTM + Sentiment-Inspired Financial ML Demo")

# -----------------------------
# Feature Engineering
# -----------------------------
def get_data():
    df = yf.download("^NSEI", period="6mo")
    df["return"] = df["Close"].pct_change()
    df["volatility"] = df["return"].rolling(20).std()
    df["hl_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["log_volume"] = np.log1p(df["Volume"])
    df = df.dropna()
    return df


# -----------------------------
# Haar Wavelet Denoising
# -----------------------------
def haar_denoise(series):
    coeffs = pywt.wavedec(series, "haar", level=1)
    coeffs[1] = np.zeros_like(coeffs[1])
    return pywt.waverec(coeffs, "haar")[:len(series)]


# -----------------------------
# Simple LSTM Model
# -----------------------------
class LSTMModel(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=32):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Next-Day Direction"):

    with st.spinner("Running inference..."):

        df = get_data()

        df["return"] = haar_denoise(df["return"].values)

        features = df[["return", "volatility", "hl_range", "log_volume"]].values[-20:]

        X = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        model = LSTMModel()
        model.eval()

        with torch.no_grad():
            prob = torch.sigmoid(model(X)).item()

        st.metric("Probability of Up Move", f"{prob:.2%}")

        if prob > 0.55:
            st.success("Bullish Signal")
        elif prob < 0.45:
            st.error("Bearish Signal")
        else:
            st.warning("Neutral Signal")
