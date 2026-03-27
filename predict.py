import pandas as pd
import numpy as np
import torch
import joblib
from train_lstm import TrafficLSTM  # reuse the model definition

def load_xgboost():
    model = joblib.load('xgboost_model.pkl')
    return model

def load_lstm(input_dim=5):   # adjust based on features used
    model = TrafficLSTM(input_dim=input_dim, hidden_dim=64, num_layers=2, output_dim=1)
    model.load_state_dict(torch.load('lstm_model.pth'))
    model.eval()
    scaler = joblib.load('lstm_scaler.pkl')
    return model, scaler

def predict_xgboost(df_row):
    # Expects a row with all feature columns used in XGBoost training
    model = load_xgboost()
    features = ['hour', 'day_of_week', 'weather_temp', 'weather_precip',
                'volume', 'speed_lag_1', 'speed_lag_2', 'speed_lag_3',
                'rolling_mean_3']
    X = df_row[features].values.reshape(1,-1)
    return model.predict(X)[0]

def predict_lstm(df_window, scaler):
    # df_window should contain 6 rows (seq_len) of raw features
    model, _ = load_lstm()
    feature_cols = ['hour', 'day_of_week', 'weather_temp', 'weather_precip',
                    'volume', 'speed']
    X = scaler.transform(df_window[feature_cols].values)   # shape (seq_len, n_features)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0)  # batch dimension
    with torch.no_grad():
        pred = model(X_tensor).item()
    return pred

if __name__ == '__main__':
    # Example: load last 6 rows from data and predict next speed
    df = pd.read_csv('traffic_data.csv', parse_dates=['timestamp'])
    seq_len = 6
    last_window = df.iloc[-seq_len:].copy()
    # For XGBoost we need lag features – we can compute them (simplified)
    # Here we just demonstrate loading and printing model info
    print("XGBoost model loaded:", load_xgboost())
    print("LSTM model and scaler loaded:", load_lstm())
    # Optional: actual prediction
    # pred = predict_lstm(last_window, scaler)
    # print("LSTM prediction for next step:", pred)
