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

---

## 🏗️ Architecture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                    SMART CITY TRAFFIC PREDICTION                        ║
║                         SYSTEM ARCHITECTURE                             ║
╠══════════════════════════════════════════════════════════════════════════╣
║                                                                          ║
║   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐                  ║
║   │  📁 Traffic  │   │ 🌦️ Weather  │   │ 📅 Events   │                  ║
║   │     CSV     │   │     API     │   │    Feed     │                  ║
║   └──────┬──────┘   └──────┬──────┘   └──────┬──────┘                  ║
║          │                 │                 │                          ║
║          └─────────────────┼─────────────────┘                          ║
║                            ▼                                             ║
║               ┌────────────────────────┐                                 ║
║               │   🔄  Data Ingestion   │                                 ║
║               │    generate_data.py    │                                 ║
║               └────────────┬───────────┘                                 ║
║                            │                                             ║
║                            ▼                                             ║
║               ┌────────────────────────┐                                 ║
║               │   ⚙️  Feature Store    │                                 ║
║               │                        │                                 ║
║               │  • Lag features        │                                 ║
║               │  • Rolling windows     │                                 ║
║               │  • Time embeddings     │                                 ║
║               │  • Seasonal encoding   │                                 ║
║               └────────────┬───────────┘                                 ║
║                            │                                             ║
║            ┌───────────────┴────────────────┐                            ║
║            ▼                                ▼                            ║
║   ┌─────────────────┐            ┌─────────────────┐                     ║
║   │  ⚡  XGBoost    │            │   🧠   LSTM     │                     ║
║   │    Training     │            │    Training     │                     ║
║   │                 │            │                 │                     ║
║   │ train_          │            │ train_          │                     ║
║   │ xgboost.py      │            │ lstm.py         │                     ║
║   └────────┬────────┘            └────────┬────────┘                     ║
║            │                              │                              ║
║            ▼                              ▼                              ║
║   ┌─────────────────┐            ┌─────────────────┐                     ║
║   │ xgboost_model   │            │ lstm_model.pth  │                     ║
║   │     .pkl        │            │   + scaler      │                     ║
║   └────────┬────────┘            └────────┬────────┘                     ║
║            └───────────────┬──────────────┘                              ║
║                            ▼                                             ║
║               ┌────────────────────────┐                                 ║
║               │   🚀  FastAPI Server   │                                 ║
║               │   run_dashboard.py     │                                 ║
║               │      port: 8000        │                                 ║
║               └────────────┬───────────┘                                 ║
║                            │                                             ║
║          ┌─────────────────┼─────────────────┐                           ║
║          ▼                 ▼                 ▼                           ║
║   ┌─────────────┐   ┌───────────────┐   ┌──────────┐                    ║
║   │ POST        │   │ POST          │   │ GET      │                    ║
║   │ /predict/   │   │ /predict/     │   │ /health  │                    ║
║   │ xgboost     │   │ lstm          │   │          │                    ║
║   └──────┬──────┘   └──────┬────────┘   └──────────┘                    ║
║          └─────────────────┘                                             ║
║                       ▲                                                  ║
║          ┌────────────┴─────────────┐                                    ║
║          │                          │                                    ║
║   ┌──────────────┐       ┌──────────────────┐                            ║
║   │ 🖥️ Dashboard │       │ 🔌 External Apps │                            ║
║   │  port 8001   │       │                  │                            ║
║   └──────────────┘       └──────────────────┘                            ║
║                                                                          ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## 🔄 Workflow

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 1 ▸ Data Generation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  $ python generate_data.py

  Outputs: traffic_data.csv
  ├── 60 days of 5-minute interval records
  ├── Columns: timestamp, speed, volume, weather_temp, weather_precip
  └── ~17,280 rows ready for training


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 2 ▸ Feature Engineering & Model Training
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ┌──────────────────────────┬───────────────────────────────┐
  │     XGBoost Path         │         LSTM Path             │
  ├──────────────────────────┼───────────────────────────────┤
  │ 1. Build lag features    │ 1. Build 6-step sequences     │
  │ 2. Train regressor       │ 2. Normalize with scaler      │
  │ 3. Evaluate on test set  │ 3. Train 2-layer LSTM         │
  │ 4. Save .pkl             │ 4. Save .pth + scaler         │
  └──────────────────────────┴───────────────────────────────┘

  $ python train_xgboost.py    →   xgboost_model.pkl
  $ python train_lstm.py       →   lstm_model.pth + scaler


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 3 ▸ Start the API Server
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  $ python run_dashboard.py

  ✓ Loads XGBoost model
  ✓ Loads LSTM model + scaler
  ✓ FastAPI live at  http://localhost:8000
  ✓ Docs live at    http://localhost:8000/docs


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 4 ▸ Launch the Dashboard
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  $ python -m http.server 8001

  Open → http://localhost:8001/dashboard.html


━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  STEP 5 ▸ Predict!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Fill inputs  →  Click Predict  →  See speed forecast (km/h)
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

### ⚡ XGBoost — Tabular Gradient Boosting

```
INPUT FEATURES (9 total)
────────────────────────────────────────────────────────────
  Temporal      │ hour, day_of_week
  Weather       │ weather_temp, weather_precip
  Traffic       │ volume
  Lag Features  │ speed_lag_1, speed_lag_2, speed_lag_3
  Aggregates    │ rolling_mean_3
────────────────────────────────────────────────────────────
OUTPUT          │ predicted_speed  (km/h, float)
OBJECTIVE       │ regression:squarederror
ARTIFACT        │ xgboost_model.pkl
────────────────────────────────────────────────────────────
WHY XGBOOST?
  ✓ Fast inference at prediction time
  ✓ Handles tabular data extremely well
  ✓ Naturally interpretable feature importance
  ✓ Robust to missing values and outliers
```

### 🧠 LSTM — Deep Sequence Learning

```
ARCHITECTURE
────────────────────────────────────────────────────────────
  Input Shape   │ (batch_size, 6 timesteps, 6 features)
                │
  LSTM Layer 1  │ 64 hidden units, return_sequences=True
  LSTM Layer 2  │ 64 hidden units
  Dropout       │ 0.2
  Dense Output  │ 1 unit (speed prediction)
────────────────────────────────────────────────────────────
OUTPUT          │ predicted_speed  (km/h, float)
FRAMEWORK       │ PyTorch 2.0+
ARTIFACTS       │ lstm_model.pth + scaler
────────────────────────────────────────────────────────────
WHY LSTM?
  ✓ Captures temporal dependencies across timesteps
  ✓ Learns patterns like rush-hour cycles
  ✓ Handles long-range seasonality
  ✓ Generalises well to unseen sequences
```

---

## 🖥️ Dashboard

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
│                                                         │
│                     [ PREDICT ]                         │
│                                                         │
│            Predicted Speed:  43.2 km/h  🟢              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│              🧠  LSTM Prediction                         │
│─────────────────────────────────────────────────────────│
│  Paste 6-step JSON sequence:                            │
│  ┌───────────────────────────────────────────────────┐  │
│  │ [{"hour":17,"day_of_week":3,"weather_temp":22.5,  │  │
│  │   "weather_precip":0,"volume":180,"speed":45},    │  │
│  │  ...5 more timesteps...]                          │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│                     [ PREDICT ]                         │
│                                                         │
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

