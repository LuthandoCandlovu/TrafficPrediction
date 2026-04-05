<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=200&section=header&text=Smart%20City%20Traffic%20Prediction&fontSize=36&fontColor=fff&animation=twinkling&fontAlignY=40&desc=ML-Powered%20Real-Time%20Congestion%20Forecasting&descAlignY=62&descSize=16" width="100%"/>

<br/>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=20&pause=1000&color=00D4FF&center=true&vCenter=true&width=700&lines=LSTM+%2B+XGBoost+Hybrid+Traffic+Forecasting;FastAPI+%7C+PyTorch+%7C+Interactive+Dashboard;Real-Time+Predictions+with+Weather+Integration;Built+for+Smart+Cities+%F0%9F%8C%86" alt="Typing SVG"/>

<br/><br/>

[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)](https://xgboost.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-F7DF1E?style=for-the-badge)](https://opensource.org/licenses/MIT)

<br/>

[![Stars](https://img.shields.io/github/stars/LuthandoCandlovu/TrafficPrediction?style=social)](https://github.com/LuthandoCandlovu/TrafficPrediction)
[![Forks](https://img.shields.io/github/forks/LuthandoCandlovu/TrafficPrediction?style=social)](https://github.com/LuthandoCandlovu/TrafficPrediction)
[![Issues](https://img.shields.io/github/issues/LuthandoCandlovu/TrafficPrediction?color=red)](https://github.com/LuthandoCandlovu/TrafficPrediction/issues)

</div>

---

## 🌆 What is Smart City Traffic Prediction?

<div align="center">
  <img src="https://media.giphy.com/media/l0HlHFRbmaZtBRhXG/giphy.gif" width="500" alt="Smart City Traffic"/>
</div>

<br/>

> **Smart City Traffic Prediction** is an end-to-end machine learning system that forecasts urban traffic speed in real time — helping city planners, commuters, and smart city platforms make faster, data-driven decisions about road congestion.

Urban traffic congestion costs billions in lost productivity every year. This project tackles that problem by combining two powerful ML models — **XGBoost** (lightning-fast tabular predictions) and **LSTM** (deep learning for time-series patterns) — into a hybrid forecasting engine that predicts the average traffic speed for the next 5-minute interval at any road sensor.

### 🔑 How it works at a glance:

- 📊 **Ingests** 60 days of traffic data (speed, volume) fused with weather signals (temperature, precipitation)
- ⚙️ **Engineers** temporal lag features, rolling windows, and time embeddings
- 🧠 **Trains** an XGBoost regressor and a 2-layer LSTM neural network in parallel
- 🚀 **Serves** both models through a production-ready **FastAPI** REST API
- 🖥️ **Visualises** live predictions through an interactive **HTML dashboard** — zero setup required

Whether you're a researcher exploring hybrid ML architectures, a developer building smart city integrations, or a data scientist benchmarking traffic models — this project gives you a fully working, end-to-end foundation.

---

## 📌 Table of Contents

<details>
<summary>Click to expand</summary>

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🔄 Workflow](#-workflow)
- [🚀 Quick Start](#-quick-start)
- [📊 API Reference](#-api-reference)
- [🧠 ML Models](#-ml-models)
- [🖥️ Dashboard](#️-dashboard)
- [🗄️ Data Sources](#️-data-sources)
- [📁 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

</details>

---

## ✨ Features

<div align="center">

| 🔮 | 🧠 | 🌦️ | 🌐 | 🖥️ | 📊 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Traffic Speed Prediction** | **Hybrid ML Models** | **Weather Integration** | **REST API** | **Live Dashboard** | **Data Generator** |
| Predicts avg speed for next 5-min interval | XGBoost + LSTM ensemble | Temp, precipitation & more | FastAPI + CORS, production-ready | HTML/JS frontend, zero setup | 60 days of synthetic data |

</div>

<div align="center">
  <img src="https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif" width="400" alt="Machine Learning in Action"/>
</div>

---

## 🏗️ Architecture

```mermaid
flowchart TD
    subgraph SOURCES["📥 Data Sources"]
        A[📁 Traffic CSV]
        B[🌦️ Weather API]
        C[📅 Events Feed]
    end

    subgraph INGESTION["🔄 Data Ingestion"]
        D[generate_data.py]
    end

    subgraph FEATURES["⚙️ Feature Store"]
        E["• Lag features
• Rolling windows
• Time embeddings
• Seasonal encoding"]
    end

    subgraph TRAINING["🏋️ Model Training"]
        direction LR
        F[train_xgboost.py]
        G[train_lstm.py]
    end

    subgraph ARTIFACTS["📦 Model Artifacts"]
        direction LR
        H[xgboost_model.pkl]
        I[lstm_model.pth + scaler]
    end

    subgraph API["🚀 FastAPI Server — port 8000"]
        J[run_dashboard.py]
        K[POST /predict/xgboost]
        L[POST /predict/lstm]
        M[GET /health]
    end

    subgraph CLIENTS["🖥️ Clients"]
        direction LR
        N[Dashboard — port 8001]
        O[External Apps]
    end

    A & B & C --> D
    D --> E
    E --> F & G
    F --> H
    G --> I
    H & I --> J
    J --> K & L & M
    K & L --> N & O

    style SOURCES fill:#1e3a5f,stroke:#00D4FF,color:#fff
    style INGESTION fill:#1a4731,stroke:#00ff88,color:#fff
    style FEATURES fill:#3d2b00,stroke:#ffaa00,color:#fff
    style TRAINING fill:#3d1a00,stroke:#ff6600,color:#fff
    style ARTIFACTS fill:#2d1a3d,stroke:#cc66ff,color:#fff
    style API fill:#1a1a3d,stroke:#6688ff,color:#fff
    style CLIENTS fill:#1a3d1a,stroke:#66ff88,color:#fff
```

---

## 🔄 Workflow

```mermaid
flowchart LR
    S1["**Step 1**
    🗄️ Generate Data
    python generate_data.py
    → traffic_data.csv
    60 days · 17 280 rows"]

    S2A["**Step 2A**
    ⚡ Train XGBoost
    python train_xgboost.py
    → xgboost_model.pkl"]

    S2B["**Step 2B**
    🧠 Train LSTM
    python train_lstm.py
    → lstm_model.pth + scaler"]

    S3["**Step 3**
    🚀 Start API
    python run_dashboard.py
    localhost:8000"]

    S4["**Step 4**
    🖥️ Launch Dashboard
    python -m http.server 8001
    localhost:8001/dashboard.html"]

    S5["**Step 5**
    🔮 Predict!
    Fill inputs → Click Predict
    → Speed forecast km/h"]

    S1 --> S2A & S2B
    S2A & S2B --> S3
    S3 --> S4
    S4 --> S5

    style S1 fill:#0d1b2a,stroke:#00D4FF,color:#fff
    style S2A fill:#0d1b2a,stroke:#ff6600,color:#fff
    style S2B fill:#0d1b2a,stroke:#EE4C2C,color:#fff
    style S3 fill:#0d1b2a,stroke:#009688,color:#fff
    style S4 fill:#0d1b2a,stroke:#66ff88,color:#fff
    style S5 fill:#0d1b2a,stroke:#ffaa00,color:#fff
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.8 or higher**
- Git
- Terminal (PowerShell / bash / zsh)

### Installation

**1. Clone the repository**
```bash
git clone https://github.com/LuthandoCandlovu/TrafficPrediction.git
cd TrafficPrediction
```

**2. Create & activate a virtual environment**
```bash
# Create
python -m venv venv

# Activate — macOS / Linux
source venv/bin/activate

# Activate — Windows
venv\Scripts\activate
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

> Don't have `requirements.txt`? Generate one:
> ```bash
> pip install fastapi uvicorn torch xgboost pandas scikit-learn numpy
> pip freeze > requirements.txt
> ```

**4. Generate data** *(optional — default CSV is included)*
```bash
python generate_data.py
```

**5. Train the models** *(or use the pre-trained ones in the repo)*
```bash
python train_xgboost.py
python train_lstm.py
```

**6. Run the system**

Open two terminals side by side:

```bash
# Terminal 1 — API backend
python run_dashboard.py
```

```bash
# Terminal 2 — Frontend server
python -m http.server 8001
```

**7. Open your browser**
```
http://localhost:8001/dashboard.html
```

---

## 📊 API Reference

### `POST /predict/xgboost`

> Predicts traffic speed from 9 tabular features.

**Request**
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

**Response**
```json
{
  "predicted_speed": 43.2
}
```

---

### `POST /predict/lstm`

> Predicts traffic speed from a sequence of 6 consecutive timesteps.

**Request**
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

**Response**
```json
{
  "predicted_speed": 40.1
}
```

---

## 🧠 ML Models

<div align="center">
  <img src="https://media.giphy.com/media/3oKIPEqDGUULpEU0aQ/giphy.gif" width="420" alt="Data Visualisation"/>
</div>

### ⚡ XGBoost — Tabular Gradient Boosting

```mermaid
flowchart LR
    subgraph INPUT["Input Features (9 total)"]
        direction TB
        T["🕐 Temporal
        hour · day_of_week"]
        W["🌦️ Weather
        temp · precipitation"]
        V["🚗 Traffic
        volume"]
        L["📉 Lag Features
        speed_lag_1 · lag_2 · lag_3"]
        R["📊 Aggregates
        rolling_mean_3"]
    end

    XGB["⚡ XGBoost Regressor
    objective: regression:squarederror
    artifact: xgboost_model.pkl"]

    OUT["🔮 predicted_speed
    (km/h, float)"]

    T & W & V & L & R --> XGB --> OUT

    style INPUT fill:#1a2a1a,stroke:#ff6600,color:#fff
    style XGB fill:#3d2000,stroke:#ff6600,color:#fff
    style OUT fill:#0d1b0d,stroke:#66ff88,color:#fff
```

### 🧠 LSTM — Deep Sequence Learning

```mermaid
flowchart LR
    subgraph SEQ["Input Sequence — 6 Timesteps"]
        direction TB
        T1["t-5: hour · dow · temp · precip · vol · speed"]
        T2["t-4: ..."]
        T3["t-3: ..."]
        T4["t-2: ..."]
        T5["t-1: ..."]
        T6["t:   hour · dow · temp · precip · vol · speed"]
    end

    subgraph NET["LSTM Network (PyTorch 2.0+)"]
        direction TB
        L1["LSTM Layer 1 — 64 hidden units"]
        L2["LSTM Layer 2 — 64 hidden units"]
        D["Dropout — 0.2"]
        FC["Dense Output — 1 unit"]
    end

    OUT["🔮 predicted_speed
    (km/h, float)"]

    T1 & T2 & T3 & T4 & T5 & T6 --> L1
    L1 --> L2 --> D --> FC --> OUT

    style SEQ fill:#1a0d2a,stroke:#EE4C2C,color:#fff
    style NET fill:#1a0d2a,stroke:#cc66ff,color:#fff
    style OUT fill:#0d1b0d,stroke:#66ff88,color:#fff
```

---

## 🖥️ Dashboard

<div align="center">
  <img src="https://media.giphy.com/media/077i6AULCXc0FKTj9s/giphy.gif" width="460" alt="Live Dashboard Preview"/>

  <br/>

  > 💡 **Pro tip:** Replace this GIF with a screen recording of your own running dashboard!
  > Use [ScreenToGif](https://www.screentogif.com/) on Windows or [Kap](https://getkap.co/) on Mac, then drag-and-drop it into this README on GitHub.
</div>

<br/>

The dashboard serves two interactive prediction cards at `http://localhost:8001/dashboard.html`:

```
┌─────────────────────────────────────────────────────────┐
│              ⚡  XGBoost Prediction                      │
│─────────────────────────────────────────────────────────│
│  Hour          [ 17  ]    Day of Week    [  3  ]         │
│  Temperature   [ 22.5]    Precipitation  [ 0.0 ]         │
│  Volume        [ 180 ]    Speed Lag 1    [  45 ]         │
│  Speed Lag 2   [  48 ]    Speed Lag 3    [  52 ]         │
│  Rolling Mean  [48.33]                                   │
│                     [ PREDICT ]                         │
│            Predicted Speed:  43.2 km/h  🟢              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              🧠  LSTM Prediction                         │
│─────────────────────────────────────────────────────────│
│  Paste 6-step JSON sequence:                            │
│  [ {"hour":17, "day_of_week":3, ...} × 6 steps ]       │
│                     [ PREDICT ]                         │
│            Predicted Speed:  40.1 km/h  🟡              │
└─────────────────────────────────────────────────────────┘
```

---

## 🗄️ Data Sources

| Status | Source | Description |
|:---:|---|---|
| 🟢 **Active** | Synthetic Generator (`generate_data.py`) | 60 days · 5-min intervals · speed, volume, weather |
| 🔵 **Planned** | [Caltrans PeMS](https://pems.dot.ca.gov/) | California real-time loop detector data |
| 🔵 **Planned** | [NYC OpenData](https://data.cityofnewyork.us/) | Real-Time Traffic Speed Data API |
| 🔵 **Planned** | [OpenWeatherMap API](https://openweathermap.org/api) | Live weather integration |

---

## 📁 Project Structure

```
TrafficPrediction/
│
├── 📄 generate_data.py       # Synthetic traffic + weather data generator
├── 📄 train_xgboost.py       # XGBoost feature engineering & training
├── 📄 train_lstm.py          # LSTM sequence model training (PyTorch)
├── 📄 run_dashboard.py       # FastAPI server (loads both models)
├── 📄 dashboard.html         # Interactive prediction frontend
│
├── 📦 xgboost_model.pkl      # Saved XGBoost model
├── 📦 lstm_model.pth         # Saved LSTM weights
├── 📦 scaler                 # Feature scaler for LSTM
│
├── 📊 traffic_data.csv       # Generated / real traffic dataset
├── 📋 requirements.txt       # Python dependencies
└── 📄 README.md              # You are here
```

---

## 🤝 Contributing

Contributions are very welcome! Here's how:

```bash
# 1. Fork the repo on GitHub

# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/TrafficPrediction.git
cd TrafficPrediction

# 3. Create your feature branch
git checkout -b feature/your-amazing-feature

# 4. Make your changes and commit
git add .
git commit -m "feat: add your amazing feature"

# 5. Push to your fork
git push origin feature/your-amazing-feature

# 6. Open a Pull Request on GitHub 🎉
```

**Ideas welcome:**
- 🔌 Real API integrations (Caltrans, NYC OpenData)
- 🗺️ Map-based congestion visualisation
- 📈 Model performance dashboard (MAE, RMSE charts)
- 🔁 Auto-retraining pipeline
- 🐳 Docker & deployment configuration

---

## 📄 License

Distributed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

<div align="center">

### 🙌 Built With

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=flat-square&logoColor=white)](https://xgboost.ai)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)

<br/>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

**Built with ❤️ by [Luthando Candlovu](https://github.com/LuthandoCandlovu)**

*⭐ Star this repo if you found it useful!*

</div>

