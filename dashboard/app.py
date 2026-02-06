import streamlit as st
import requests

st.title("FinSight AI – NIFTY 50 Intelligence Dashboard")

API_BASE = "http://127.0.0.1:8001"

# ------------------------
# Health Check
# ------------------------
if st.button("Check API Health"):
    try:
        r = requests.get(f"{API_BASE}/health")
        st.write(r.json())
    except Exception as e:
        st.error(f"API Error: {e}")

# ------------------------
# Prediction
# ------------------------
if st.button("Predict Next-Day Direction"):
    try:
        r = requests.post(f"{API_BASE}/predict")
        result = r.json()

        if "probability_up" in result:
            prob = result["probability_up"]

            st.metric("Probability of Up Move", f"{prob:.2%}")

            if prob > 0.55:
                st.success("Bullish Signal")
            elif prob < 0.45:
                st.error("Bearish Signal")
            else:
                st.warning("Neutral Signal")
        else:
            st.error(f"Unexpected response: {result}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
