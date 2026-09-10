# supply-chain
project link # supply-chain

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://supply-chain-dhru.streamlit.app/)

An end-to-end supply chain machine learning project.

## 📌 Key Project Features

### 1. Dual Machine Learning Pipeline
- **Classification Modeling:**
  - Identifies critical supply chain risks such as late delivery probability, fraud transaction detection, or order cancellation risks.
  - Utilizes pre-trained algorithms to classify operational statuses in real time.
- **Regression Modeling:**
  - Forecasts continuous numerical targets such as estimated shipping/delivery days, product demand, and sales/profit margins.
  - Helps businesses proactively manage stock levels and logistics schedules.

### 2. Automated Data Preprocessing & Scaling
- **Standardized Transformations:** Pre-trained standard scalers (`scaler_classification.pkl`, `scaler_regression.pkl`) ensure input features are normalized dynamically before inference.
- **Categorical Handling:** Encodes operational variables (shipping modes, customer segments, product categories, and geographic regions) for consistent predictions.

### 3. Interactive Web Dashboard (Streamlit)
- **Live User Interface:** Hosted on Streamlit Cloud for interactive parameter inputs and rapid scenario testing.
- **Instant Inference:** Delivers real-time predictions for both classification and regression tasks upon submitting supply chain inputs.
- **Clean Visualizations:** Displays probability scores and forecasted numerical metrics in a structured, readable layout.

### 4. Production-Ready Deployment
- **Model Serialization:** Uses lightweight, production-ready pickle models for low-latency inference.
- **Modular Architecture:** Self-contained structure (`app.py`, dependency specifications, and trained artifacts) ensuring simple maintenance and cloud portability.
