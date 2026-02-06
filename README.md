# fintech_ai

# 🚀 FinSight AI – NIFTY 50 Financial Intelligence Platform

> 🧠 Production-Grade Machine Learning System for Sentiment-Aware Index Direction Prediction  
> 📈 Built with Deep Learning, Walk-Forward Validation, and Microservice Deployment  

---

![Python](https://img.shields.io/badge/Python-3.10-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-DeepLearning-red)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-orange)
![Docker](https://img.shields.io/badge/Docker-Containerized-blue)
![MLflow](https://img.shields.io/badge/MLflow-ExperimentTracking-purple)

---

# 📌 Overview

**FinSight AI** is a production-structured financial machine learning platform designed to predict the **next-day direction of the NIFTY 50 Index** using:

- 📊 Market-derived features  
- 📰 Financial news sentiment (FinBERT)  
- 🌊 Haar wavelet denoising  
- 🧠 LSTM deep learning  
- 🔁 Walk-forward retraining  
- 📉 Volatility-targeted backtesting  

This project demonstrates **real-world ML engineering practices**, not just notebook experimentation.

---

# 🏗 System Architecture

```
Market Data (yfinance)
        │
        ▼
Feature Engineering
        │
        ▼
Haar Wavelet Denoising
        │
        ▼
FinBERT Sentiment Scoring
        │
        ▼
LSTM Classification Model
        │
        ▼
Walk-Forward Retraining
        │
        ▼
Volatility-Targeted Backtesting
        │
        ▼
FastAPI Inference API
        │
        ▼
Streamlit Dashboard
```

---

# 🧰 Tech Stack

## 🧠 Machine Learning
- Python 3.10
- PyTorch
- Transformers (FinBERT)
- Scikit-learn
- PyWavelets

## 📈 Financial Modeling
- yfinance (NIFTY 50 Index)
- Rolling volatility modeling
- Walk-forward validation
- Sharpe ratio backtesting

## 🌐 Backend
- FastAPI
- Pydantic
- MLflow (experiment tracking)

## 🖥 Frontend
- Streamlit Dashboard

## 🐳 DevOps
- Docker
- Docker Compose
- Modular microservice architecture

---

# 📊 Modeling Approach

## 🎯 Objective

Predict:

```
P(Next-Day NIFTY 50 Close > Today Close)
```

(Binary classification)

---

## 📈 Features Used

- Daily return
- Rolling volatility (20-day)
- High-Low range
- Log volume
- News sentiment score (FinBERT)
- Wavelet-denoised price signal

---

## 🔁 Validation Strategy

Instead of a static train-test split:

- 5-Year Rolling Training Window  
- 1-Year Out-of-Sample Testing  
- Walk-Forward Retraining  

This simulates real-world deployment and prevents regime leakage.

---

# 📉 Backtesting Strategy

Signal Logic:

```
If P(Up) > 0.55 → Long
Else → Cash
```

Position sizing:

```
Volatility-targeted exposure
```

Evaluation Metrics:

- Accuracy
- AUC
- Sharpe Ratio
- Strategy vs Market equity curve

---

# 🚀 Running Locally

## 1️⃣ Clone Repository

```bash
git clone https://github.com/Harshithpatali/fintech_ai.git
cd fintech_ai
```

---

## 2️⃣ Create Virtual Environment

```bash
python -m venv .venv
.\.venv\Scripts\activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4️⃣ Run Walk-Forward Training

```bash
python run_training.py
```

---

## 5️⃣ Run FastAPI Backend

```bash
uvicorn app.api.main:app --reload --port 8001
```

Swagger UI:

```
http://127.0.0.1:8001/docs
```

---

## 6️⃣ Run Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

Open:

```
http://localhost:8501
```

---

# 🐳 Docker Deployment

## Build & Run

```bash
docker-compose build
docker-compose up
```

### Services

- Backend → http://localhost:8001/docs
- Frontend → http://localhost:8501

---

# 📂 Project Structure

```
fintech_ai/
│
├── app/
│   ├── api/              # FastAPI service
│   ├── core/             # Training, walk-forward, backtesting
│   ├── models/           # LSTM model
│   ├── pipelines/        # Data ingestion pipeline
│   ├── services/         # Market & news services
│   └── config.py
│
├── dashboard/            # Streamlit UI
├── docker/               # Dockerfiles
├── docker-compose.yml
├── run_training.py
├── requirements.txt
└── README.md
```

---

# 📈 Example Walk-Forward Output

```
Walk-Forward Accuracy: ~55%
Walk-Forward AUC: ~0.58
Strategy Sharpe: ~0.40
Market Sharpe: ~1.19
```

This reflects:

- Realistic performance
- No data leakage
- Regime-aware validation
- Production-level evaluation

---

# 🎯 Engineering Highlights

- Clean modular architecture
- Config-driven hyperparameters
- Time-series safe validation
- No data leakage
- Gradient clipping
- MLflow experiment tracking
- REST API serving
- Containerized microservices

---

# 🧠 Why This Project Matters

This project demonstrates the ability to:

- Build production-grade ML systems
- Handle noisy financial time-series
- Integrate NLP sentiment with market data
- Implement walk-forward validation
- Design deployable microservices
- Apply quantitative risk-adjusted evaluation

---

# 👨‍💻 Author

**Harshith Devraj**  
Machine Learning Engineer | Quant-Focused AI Developer  

---

# ⭐ If You Like This Project

Star ⭐ the repository  
Fork 🍴 it  
Build on it 🚀  

---
