<div align="center">

# 🚦 Smart City Traffic Prediction System

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&pause=1000&color=00D4FF&center=true&vCenter=true&width=600&lines=Real-Time+Traffic+Congestion+Prediction;LSTM+%2B+XGBoost+Hybrid+Models;FastAPI+%7C+PyTorch+%7C+Interactive+Dashboard" alt="Typing SVG" />

<br/>

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br/>

> A machine learning system that predicts short‑term traffic congestion using **LSTM** and **XGBoost**, integrating weather and event data. Built with **FastAPI** for real‑time predictions and an interactive dashboard.

<br/>

</div>

---

## 📌 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🔄 Workflow](#-workflow)
- [🚀 Quick Start](#-quick-start)
- [📊 API Endpoints](#-api-endpoints)
- [📈 Models](#-models)
- [🧪 Testing with Dashboard](#-testing-with-dashboard)
- [🗄️ Data Sources](#️-data-sources)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

---

## ✨ Features

| Feature | Description |
|---|---|
| 🔮 **Traffic Speed Prediction** | Predicts average speed for the next 5-minute interval |
| 🧠 **Hybrid Models** | Combines XGBoost (baseline) and LSTM (deep learning) |
| 🌦️ **Multi-Source Data** | Integrates historical traffic, weather, and temporal features |
| 🌐 **REST API** | Built with FastAPI, ready for real-time integration |
| 🖥️ **Interactive Dashboard** | Simple HTML/JS frontend to test predictions live |
| 📊 **Synthetic Data Generator** | Generates realistic traffic datasets for development |
| 🧪 **Time-Series Features** | Rolling windows, lag variables, and seasonal encodings |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SYSTEM ARCHITECTURE                          │
└─────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
  │ Data Sources │────▶│ Data         │────▶│  Feature Store   │
  │              │     │ Ingestion    │     │                  │
  │ • Traffic CSV│     │              │     │ • Lag features   │
  │ • Weather API│     │              │     │ • Rolling stats  │
  │ • Event Feeds│     │              │     │ • Time encodings │
  └──────────────┘     └──────────────┘     └────────┬─────────┘
                                                     │
                              ┌──────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Model Training  │
                    │                  │
                    │  ┌────────────┐  │
                    │  │  XGBoost   │  │
                    │  └────────────┘  │
                    │  ┌────────────┐  │
                    │  │    LSTM    │  │
                    │  └────────────┘  │
                    └────────┬─────────┘
                             │
                    ┌────────▼─────────┐
                    │  Saved Models    │
                    │                  │
                    │ xgboost_model    │
                    │    .pkl          │
                    │ lstm_model.pth   │
                    │ scaler           │
                    └────────┬─────────┘
                             │
              ┌──────────────▼──────────────┐
              │       FastAPI Backend        │
              │        (port 8000)           │
              └──────────────┬──────────────┘
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
      ┌────────────┐  ┌────────────┐  ┌────────────┐
      │ /predict/  │  │ /predict/  │  │  /health   │
      │  xgboost  │  │    lstm    │  │            │
      └────────────┘  └────────────┘  └────────────┘
             ▲               ▲
             └───────┬───────┘
                     │
          ┌──────────┴──────────┐
          │   Dashboard (HTML)  │        ┌─────────────────┐
          │    (port 8001)      │        │  External Apps  │
          └─────────────────────┘        └─────────────────┘
```

---

## 🔄 Workflow

```
Step 1 ──────────────────────────────────────────────────────────
  📁 Data Generation
     └── generate_data.py
         Creates a CSV with 60 days of:
         traffic speed + weather + volume data

Step 2 ──────────────────────────────────────────────────────────
  🔧 Feature Engineering & Training

  ┌──────────────────────────┐   ┌──────────────────────────────┐
  │       XGBoost Path       │   │          LSTM Path           │
  │                          │   │                              │
  │  1. Create lag features  │   │  1. Create 6-step sequences  │
  │  2. Train regressor      │   │  2. Train with PyTorch       │
  │  3. Save .pkl model      │   │  3. Save .pth + scaler       │
  └──────────────────────────┘   └──────────────────────────────┘

Step 3 ──────────────────────────────────────────────────────────
  🚀 Serving Predictions
     └── run_dashboard.py
         Loads both models → starts FastAPI on port 8000

Step 4 ──────────────────────────────────────────────────────────
  🖥️  Frontend
     └── python -m http.server 8001
         Serves dashboard.html → calls API → displays predictions

Step 5 ──────────────────────────────────────────────────────────
  👤 User Interaction
     Fill input fields → Click Predict → See predicted speed (km/h)
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.8+**
- Git
- A terminal (PowerShell on Windows, bash on macOS/Linux)

### 1. Clone the Repository

```bash
git clone https://github.com/LuthandoCandlovu/TrafficPrediction.git
cd TrafficPrediction
```

### 2. Set Up Virtual Environment

```bash
# Create environment
python -m venv venv

# Activate — macOS/Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> If you don't have a `requirements.txt`, generate one with:
> ```bash
> pip freeze > requirements.txt
> ```

### 4. Generate Synthetic Data *(optional — default CSV included)*

```bash
python generate_data.py
```

### 5. Train the Models *(or use pre-trained models already in the repo)*

```bash
python train_xgboost.py
python train_lstm.py
```

### 6. Run the API & Dashboard

Open **two separate terminals**:

```bash
# Terminal 1 — FastAPI backend (port 8000)
python run_dashboard.py

# Terminal 2 — Static file server (port 8001)
python -m http.server 8001
```

Then open your browser at:

```
http://localhost:8001/dashboard.html
```

---

## 📊 API Endpoints

### `POST /predict/xgboost`

Predict speed from tabular features.

**Request Body:**
```json
{
  "features": {
    "hour": 17,
    "day_of_week": 3,
    "weather_temp": 22.5,
    "weather_precip": 0.0,
    "volume": 180,
    "speed_lag_1": 45,
    "speed_lag_2": 48,
    "speed_lag_3": 52,
    "rolling_mean_3": 48.33
  }
}
```

**Response:**
```json
{
  "predicted_speed": 43.2
}
```

---

### `POST /predict/lstm`

Predict speed from a sequence of 6 timesteps.

**Request Body:**
```json
{
  "features": [
    { "hour": 17, "day_of_week": 3, "weather_temp": 22.5, "weather_precip": 0, "volume": 180, "speed": 45 },
    { "hour": 17, "day_of_week": 3, "weather_temp": 22.5, "weather_precip": 0, "volume": 182, "speed": 44 },
    { "hour": 17, "day_of_week": 3, "weather_temp": 22.5, "weather_precip": 0, "volume": 185, "speed": 43 },
    { "hour": 17, "day_of_week": 3, "weather_temp": 22.5, "weather_precip": 0, "volume": 188, "speed": 42 },
    { "hour": 17, "day_of_week": 3, "weather_temp": 22.5, "weather_precip": 0, "volume": 190, "speed": 41 },
    { "hour": 17, "day_of_week": 3, "weather_temp": 22.5, "weather_precip": 0, "volume": 192, "speed": 40 }
  ]
}
```

**Response:**
```json
{
  "predicted_speed": 40.1
}
```

---

## 📈 Models

### ⚡ XGBoost

```
Input Features (9):
  ├── hour            — time of day
  ├── day_of_week     — 0 (Mon) → 6 (Sun)
  ├── weather_temp    — °C
  ├── weather_precip  — mm
  ├── volume          — vehicles/interval
  ├── speed_lag_1     — previous interval
  ├── speed_lag_2     — 2 intervals ago
  ├── speed_lag_3     — 3 intervals ago
  └── rolling_mean_3  — 3-step rolling average

Objective  : regression (speed in km/h)
Strength   : fast, interpretable, handles tabular data well
Output     : xgboost_model.pkl
```

### 🧠 LSTM (Long Short-Term Memory)

```
Input Shape : (batch_size, 6 timesteps, 6 features)

Architecture:
  ┌─────────────────────────────┐
  │  LSTM Layer 1 (64 units)    │
  │  LSTM Layer 2 (64 units)    │
  │  Dropout                    │
  │  Dense Output Layer (1)     │
  └─────────────────────────────┘

Strength  : captures temporal dependencies and seasonality
Output    : lstm_model.pth + scaler
```

---

## 🧪 Testing with Dashboard

The dashboard provides **two prediction cards**:

```
┌─────────────────────────────────────────────┐
│         XGBoost Prediction Card             │
│                                             │
│  Manually enter feature values:             │
│  Hour, Day, Temperature, Volume, Lags...    │
│                                             │
│  [ PREDICT ]  →  Speed: 43.2 km/h          │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│           LSTM Prediction Card              │
│                                             │
│  Paste a 6-step JSON sequence               │
│  (hour, day, weather, volume, speed)        │
│                                             │
│  [ PREDICT ]  →  Speed: 40.1 km/h          │
└─────────────────────────────────────────────┘
```

---

## 🗄️ Data Sources

| Source | Status | Description |
|---|---|---|
| 🟢 Synthetic Data | **Active** | Generated via `generate_data.py` — 60 days of traffic, weather & volume |
| 🔵 Caltrans PeMS | Planned | California Performance Measurement System |
| 🔵 NYC OpenData | Planned | Real-Time Traffic Speed Data |
| 🔵 OpenWeatherMap | Planned | Live weather API integration |

---

## 🤝 Contributing

Contributions are welcome! Here's how to get started:

```bash
# 1. Fork the repository on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/TrafficPrediction.git

# 3. Create a feature branch
git checkout -b feature/amazing-feature

# 4. Make your changes and commit
git commit -m "Add amazing feature"

# 5. Push to your fork
git push origin feature/amazing-feature

# 6. Open a Pull Request on GitHub
```

Please make sure your code follows the existing style and includes relevant tests.

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.

---

<div align="center">

### 🙌 Acknowledgements

[![FastAPI](https://img.shields.io/badge/FastAPI-Modern_Web_Framework-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=flat-square&logo=pytorch)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-Gradient_Boosting-FF6600?style=flat-square)](https://xgboost.ai)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Manipulation-150458?style=flat-square&logo=pandas)](https://pandas.pydata.org)

<br/>

Built with ❤️ by **Luthando Candlovu**

</div>

