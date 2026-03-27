🚦 Smart City Traffic Prediction System
<p align="center"> <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python"/> <img src="https://img.shields.io/badge/FastAPI-Backend-green?style=for-the-badge&logo=fastapi"/> <img src="https://img.shields.io/badge/PyTorch-DeepLearning-red?style=for-the-badge&logo=pytorch"/> <img src="https://img.shields.io/badge/XGBoost-ML-orange?style=for-the-badge"/> <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/> </p> <p align="center"> <b>AI-powered traffic congestion prediction using LSTM & XGBoost with real-time API deployment.</b> </p>
🎥 Demo Preview
<p align="center"> <img src="https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExb3Z2c3R5Y2p5eDR1dTN4a3k3bTR3dXo4eXZtY3N5Z3BxMWN5dzF0NSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3o7aD2saalBwwftBIY/giphy.gif" width="700"/> </p>

Replace this GIF later with a real dashboard screen recording (recommended 🚀).

✨ Features
🔮 Traffic Speed Prediction
Predicts next 5-minute traffic speed using historical signals.
🧠 Hybrid Machine Learning
XGBoost → tabular prediction baseline
LSTM → temporal sequence learning
🌐 Real-Time REST API
Built with FastAPI for production-ready inference.
🖥️ Interactive Dashboard
HTML + JavaScript frontend for live testing.
📊 Synthetic Data Generator
Quickly simulate realistic traffic datasets.
⏱️ Time-Series Engineering
Lag features
Rolling averages
Seasonal encodings
🏗️ System Architecture
graph TD
    A[Traffic & Weather Data] --> B[Data Ingestion]
    B --> C[Feature Engineering]
    C --> D[Model Training]
    D --> E[Saved Models]
    E --> F[FastAPI Backend]
    F --> G[Prediction API]
    H[Dashboard UI] --> F
    I[External Applications] --> F
Components
Layer	Description
Data Sources	Synthetic CSV or real APIs
Feature Store	Lag + rolling features
Models	XGBoost & LSTM
Backend	FastAPI REST API
Frontend	Interactive dashboard
🔄 Workflow
1️⃣ Data Generation
python generate_data.py

Creates 60 days of traffic + weather data.

2️⃣ Model Training

XGBoost

python train_xgboost.py

LSTM

python train_lstm.py

Outputs:

xgboost_model.pkl
lstm_model.pth
scaler.pkl
3️⃣ Run Prediction System

Open two terminals.

Terminal 1 — API Server
python run_dashboard.py
Terminal 2 — Dashboard
python -m http.server 8001

Open:

http://localhost:8001/dashboard.html
🚀 Quick Start
Prerequisites
Python 3.8+
Git
Terminal / PowerShell
Clone Repository
git clone https://github.com/LuthandoCandlovu/TrafficPrediction.git
cd TrafficPrediction
Create Virtual Environment
python -m venv venv

Activate:

Windows

venv\Scripts\activate

Mac/Linux

source venv/bin/activate
Install Dependencies
pip install -r requirements.txt

If missing:

pip freeze > requirements.txt
📊 API Endpoints
🔹 XGBoost Prediction

POST /predict/xgboost

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

Response:

{
  "predicted_speed": 43.2
}
🔹 LSTM Prediction

POST /predict/lstm

Input → 6 time steps sequence.

{
  "features": [ ... ]
}

Response:

{
  "predicted_speed": 40.1
}
📈 Models
🌳 XGBoost
Handles structured tabular data efficiently
Fast inference
Interpretable features
🧠 LSTM (PyTorch)

Architecture:

2 × LSTM Layers (64 units)
↓
Dropout
↓
Dense Output Layer

Captures:

temporal dependencies
traffic seasonality
trend patterns
🧪 Dashboard

The dashboard includes:

✅ Manual feature input
✅ LSTM sequence testing
✅ Instant predictions

<p align="center"> <img src="https://via.placeholder.com/800x400?text=Traffic+Dashboard+Preview"/> </p>
🗄️ Data Sources

Current:

Synthetic dataset generator

Planned integrations:

Caltrans Performance Measurement System
NYC OpenData Traffic API
OpenWeatherMap API
📂 Project Structure
TrafficPrediction/
│
├── models/
├── api/
├── dashboard/
├── generate_data.py
├── train_xgboost.py
├── train_lstm.py
├── run_dashboard.py
└── requirements.txt
🤝 Contributing
git fork
git checkout -b feature/new-feature
git commit -m "Add feature"
git push origin feature/new-feature

Then open a Pull Request 🚀

📄 License

Distributed under the MIT License.

👨‍💻 Author

Luthando Candlovu

🎓 Computer Science (Honours)
🤖 Machine Learning • AI • Smart Cities • Cybersecurity

<p align="center"> ⭐ If you like this project, give it a star! </p>
