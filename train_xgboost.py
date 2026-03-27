import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import joblib

def create_features(df, target_col='speed', lookback=6):
    """
    Create lag features and rolling statistics.
    """
    df = df.sort_values('timestamp').copy()
    for lag in range(1, lookback+1):
        df[f'speed_lag_{lag}'] = df[target_col].shift(lag)
    # Rolling mean of last 3 lags
    df['rolling_mean_3'] = df[[f'speed_lag_{i}' for i in range(1,4)]].mean(axis=1)
    # Drop rows with NaN from lags
    df.dropna(inplace=True)
    return df

def main():
    # Load data
    df = pd.read_csv('traffic_data.csv', parse_dates=['timestamp'])
    df = create_features(df)

    # Define features and target
    feature_cols = ['hour', 'day_of_week', 'weather_temp', 'weather_precip',
                    'volume', 'speed_lag_1', 'speed_lag_2', 'speed_lag_3',
                    'rolling_mean_3']
    X = df[feature_cols]
    y = df['speed']

    # Train/val split (time-based)
    split_idx = int(0.8 * len(df))
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    # Train XGBoost
    model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    print(f"XGBoost - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Save model
    joblib.dump(model, 'xgboost_model.pkl')
    print("Model saved to xgboost_model.pkl")

if __name__ == '__main__':
    main()
